use anyhow::anyhow;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::{
    io::{BufReader, Seek, Write},
    time::SystemTime,
};
use tide::{Request, Response};
use time::OffsetDateTime as Time;

// https://github.com/bohwaz/micro-gpodder-server
// https://github.com/ahgamut/rust-ape-example

fn default<T: std::default::Default>() -> T {
    T::default()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// confusingly, the event timestamps are formatted as ISO8601 date-times without an offset
/// and the offset is assumed to be UTC.
///
/// So we need some special casing to parse this correctly to a UtcDateTime, and serialize it correctly as well
mod timestamp {
    use time::{
        format_description::well_known::{iso8601, Iso8601},
        OffsetDateTime as Time, UtcOffset,
    };

    pub fn serialize<S: serde::Serializer>(t: &Time, s: S) -> Result<S::Ok, S::Error> {
        // just make sure it's UTC
        let t = t.to_offset(UtcOffset::UTC);
        let local = time::PrimitiveDateTime::new(t.date(), t.time());

        // don't print so many 0s after the second
        const ISO8601_FORMAT: iso8601::EncodedConfig = iso8601::Config::DEFAULT
            .set_time_precision(iso8601::TimePrecision::Second {
                decimal_digits: None,
            })
            .set_formatted_components(iso8601::FormattedComponents::DateTime)
            .encode();
        s.serialize_str(local.format(&Iso8601::<ISO8601_FORMAT>).unwrap().as_str())
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(dsr: D) -> Result<Time, D::Error> {
        struct Visitor;

        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Time;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("'yyyy-MM-ddTHH:mm:ss' (ISO 8601 local) timestamp")
            }

            fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<Self::Value, E> {
                if s.is_empty() {
                    return Ok(Time::UNIX_EPOCH);
                }

                match time::PrimitiveDateTime::parse(s, &Iso8601::DATE_TIME) {
                    Ok(t) => Ok(t.assume_utc()),
                    Err(_err) => Err(E::invalid_value(serde::de::Unexpected::Str(s), &self)),
                }
            }

            fn visit_u64<E: serde::de::Error>(self, n: u64) -> Result<Self::Value, E> {
                Ok(Time::UNIX_EPOCH
                    .checked_add(time::Duration::seconds(n as i64))
                    .unwrap())
            }
        }

        dsr.deserialize_str(Visitor)
    }
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
struct Podcast {
    url: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    title: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    author: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    description: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    website: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    logo_url: String,
}

// minus_opt deserializes an optional positive value, with None serialized as '-1'
fn minus_opt<'de, D: serde::Deserializer<'de>>(dsr: D) -> Result<Option<u64>, D::Error> {
    struct MinusOptVisitor;
    impl<'de> serde::de::Visitor<'de> for MinusOptVisitor {
        type Value = Option<u64>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "u64 or -1")
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }
        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(v))
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v == -1 {
                Ok(None)
            } else {
                Err(E::invalid_value(serde::de::Unexpected::Signed(v), &self))
            }
        }
    }
    dsr.deserialize_any(MinusOptVisitor)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "lowercase")]
enum Action {
    Add,
    Remove,
    Download,
    Delete,
    Play {
        #[serde(deserialize_with = "minus_opt")]
        started: Option<u64>,
        #[serde(deserialize_with = "minus_opt")]
        position: Option<u64>,
        #[serde(deserialize_with = "minus_opt")]
        total: Option<u64>,
    },
    New,
    Flattr,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Event {
    podcast: String,
    #[serde(default)]
    episode: String,
    device: String,
    #[serde(flatten)]
    action: Action,
    #[serde(with = "timestamp")]
    timestamp: Time,
}

impl std::cmp::PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for Event {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.timestamp,
            &self.podcast,
            &self.episode,
            &self.action,
            &self.device,
        )
            .cmp(&(
                other.timestamp,
                &other.podcast,
                &other.episode,
                &other.action,
                &other.device,
            ))
    }
}

#[derive(Default, Clone, PartialEq, Serialize, Deserialize)]
struct Device {
    #[serde(default)]
    id: String,
    #[serde(default)]
    caption: String,
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    subscriptions: u64,
}

#[derive(Default, Clone, Serialize, Deserialize, PartialEq)]
struct UserData {
    username: String,
    password: String,
    podcasts: Vec<Podcast>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    events: Vec<Event>,
    devices: Vec<Device>,
    devsubs: Vec<DevSubDiff>,
}

fn userdata<R>(
    username: &str,
    f: impl FnOnce(&mut UserData) -> tide::Result<R>,
) -> tide::Result<R> {
    use std::io::BufRead;
    let cfg = std::env::var("GPODRS_CONFIG_DIR").unwrap_or("./".to_string());
    let mut cfg = std::path::PathBuf::from(cfg);
    cfg.push(format!("{username}_cfg.json"));

    if !std::fs::exists(&cfg)? {
        log::info!("unable to find v2 userdata, fallback to v1");
        return userdata_v1(username, f);
    }

    let mut user_cfg = std::fs::File::options()
        .read(true)
        .write(true)
        .create(false)
        .open(&cfg)?;
    cfg.pop();
    cfg.push(format!("{username}_events.json"));
    let mut user_events = std::fs::File::options()
        .read(true)
        .write(true)
        .create(false)
        .open(&cfg)?;

    let mut userdata: UserData = serde_json::from_reader(&user_cfg)?;
    assert!(
        userdata.events.is_empty(),
        "userdata v2 does not include events in <user>_cfg.json"
    );
    for line in BufReader::new(&mut user_events).lines() {
        let event: Event = serde_json::from_str(&line?)?;
        userdata.events.push(event);
    }

    let mut data2 = userdata.clone();
    let ret = f(&mut data2)?;
    let events_pre = std::mem::take(&mut userdata.events);
    let events = std::mem::take(&mut data2.events);
    if userdata != data2 {
        user_cfg.seek(std::io::SeekFrom::Start(0))?;
        serde_json::to_writer_pretty(&user_cfg, &data2)?;
        let flen = user_cfg.stream_position()?;
        user_cfg.set_len(flen)?;
    }

    if events_pre != events {
        user_events.seek(std::io::SeekFrom::Start(0))?;
        let prefix_lines = events
            .iter()
            .zip(&events_pre)
            .take_while(|(a, b)| a == b)
            .count();
        log::debug!(
            "events share {prefix_lines} common prefix lines (out of {} total)",
            events.len()
        );

        let mut user_events = {
            let mut rdr = BufReader::new(user_events);
            // skip n lines
            for _ in 0..prefix_lines {
                rdr.skip_until(b'\n')?;
            }

            #[allow(
                clippy::seek_from_current,
                reason = "want to advance internal file as well"
            )]
            let flen = rdr.seek(std::io::SeekFrom::Current(0))?;
            let rdr = rdr.into_inner();
            rdr.set_len(flen)?;
            rdr
        };

        for event in events.into_iter().skip(prefix_lines) {
            serde_json::to_writer(&user_events, &event)?;
            writeln!(user_events)?;
        }
    }

    Ok(ret)
}

fn userdata_v1<R>(
    username: &str,
    f: impl FnOnce(&mut UserData) -> tide::Result<R>,
) -> tide::Result<R> {
    use std::io::{Seek, SeekFrom};
    let mut cfg = std::env::var("GPODRS_CONFIG_DIR").unwrap_or("./".to_string());
    if !cfg.ends_with('/') {
        cfg.push('/');
    }
    cfg.push_str(username);
    cfg.push_str(".json");
    log::info!("user config: {cfg}");
    let mut datafile = std::fs::File::options()
        .read(true)
        .write(true)
        .create(false)
        .open(cfg)?;
    let userdata: UserData = serde_json::from_reader(&datafile)?;
    let mut data2 = userdata.clone();
    let ret = f(&mut data2)?;
    if userdata != data2 {
        datafile.seek(SeekFrom::Start(0))?;
        serde_json::to_writer_pretty(&datafile, &data2)?;
        let flen = datafile.stream_position()?;
        datafile.set_len(flen)?;
    }
    Ok(ret)
}

async fn todo(mut req: Request<()>) -> tide::Result {
    let body: String = req.body_string().await?;
    log::info!("TODO: {body:?}");
    Ok(Response::builder(501).body("not implemented yet").build())
}

macro_rules! bail {
    ($status:literal, $($err:tt)*) => {
        return Err(tide::Error::new(
            $status,
            anyhow!($($err)*),
        ))
    }
}

// auth

async fn auth_login(mut req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;

    if let Some(sess_username) = req.session().get::<String>("username") {
        // check sessionid
        if sess_username != path_username {
            bail!(400, "session username is not valid for authenticated user");
        }
        return Ok("".into());
    }

    // normal auth flow
    let Some(auth_hdr) = req.header("Authorization").map(|hdrs| hdrs.last()) else {
        bail!(401, "authorization header not present",);
    };
    if !auth_hdr.as_str().starts_with("Basic ") {
        bail!(401, "authorize header is not Basic");
    }
    let auth_hdr = BASE64
        .decode(&auth_hdr.as_str()["Basic ".len()..])
        .map_err(|e| tide::Error::new(401, e))?;
    let auth_hdr = String::from_utf8(auth_hdr).map_err(|e| tide::Error::new(401, e))?;
    let Some((auth_username, auth_password)) = auth_hdr.split_once(':') else {
        bail!(401, "authorize header is not valid basic auth");
    };
    if auth_username != path_username {
        bail!(401, "login username does not match path resource");
    }

    let auth_password_hash = {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(auth_password);
        let hash = hasher.finalize();
        let mut ret = String::new();
        for b in hash {
            ret.push_str(&format!("{b:02x}"));
        }
        ret
    };
    log::info!("password hash: {auth_password_hash}");
    match userdata(auth_username, |userdata| {
        Ok(path_username.is_empty()
            || userdata.username != auth_username
            || userdata.password != auth_password_hash)
    }) {
        Err(err) => {
            if let Some(ioerr) = err.downcast_ref::<std::io::Error>() {
                if ioerr.kind() == std::io::ErrorKind::NotFound {
                    log::error!("{ioerr}");
                    bail!(401, "user {auth_username} does not exist");
                }
            }
            return Err(err);
        }
        Ok(false) => {}
        Ok(true) => {
            bail!(
                401,
                "unable to authenticate: {auth_username:?} {auth_password:?}"
            );
        }
    }

    req.session_mut().insert("username", auth_username)?;
    Ok("".into())
}

async fn auth_logout(mut req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;
    if let Some(sess_username) = req.session().get::<String>("username") {
        if path_username != sess_username {
            bail!(400, "session user does not match path resource");
        }
    }

    req.session_mut().remove("username");
    Ok("".into())
}

fn split_suffix(s: &str) -> (&str, &str) {
    match s.rfind('.') {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, ""),
    }
}

fn assert_format(f: &str) -> tide::Result<()> {
    if f != "json" {
        bail!(400, "no format specified")
    } else {
        Ok(())
    }
}

// devices

async fn list_devices(req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;
    let (path_username, format) = split_suffix(path_username);
    assert_format(format)?;
    assert_auth(&req, Some(path_username))?;
    let body_json = userdata(path_username, |userdata| {
        Ok(serde_json::to_string(&userdata.devices)?)
    })?;
    Ok(Response::builder(200)
        .body(body_json)
        .content_type("application/json")
        .build())
}

async fn update_device(mut req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;
    assert_auth(&req, Some(path_username))?;
    let path_deviceid = req.param("deviceid")?;
    let (path_deviceid, format) = split_suffix(path_deviceid);
    let path_deviceid = path_deviceid.to_string();
    assert_format(format)?;

    let mut device: Device = req.body_json().await?;
    let path_username = req.param("username")?;
    device.id = path_deviceid;
    if !device.id.is_empty() {
        userdata(path_username, |userdata| {
            if let Some(dev) = userdata.devices.iter_mut().find(|d| d.id == device.id) {
                if !device.caption.is_empty() {
                    dev.caption = device.caption;
                }
                if !device.r#type.is_empty() {
                    dev.r#type = device.r#type;
                }
                if device.subscriptions != 0 {
                    dev.subscriptions = device.subscriptions;
                }
            } else {
                userdata.devices.push(device);
            }
            Ok(())
        })?;
    }

    Ok("".into())
}

// subscriptions

async fn get_subscriptions(req: Request<()>) -> tide::Result {
    let (path_username, _path_deviceid, format) =
        match (req.param("username")?, req.param("deviceid")) {
            (u, Ok(d)) => {
                let (d, f) = split_suffix(d);
                (u, Some(d), f)
            }
            (u, Err(_)) => {
                let (u, f) = split_suffix(u);
                (u, None, f)
            }
        };
    assert_format(format)?;
    assert_auth(&req, Some(path_username))?;

    let subs = userdata(path_username, |userdata| {
        Ok(serde_json::to_string(&userdata.podcasts)?)
    })?;
    Ok(Response::builder(200)
        .content_type("application/json")
        .body(subs)
        .build())
}

fn assert_auth<T>(req: &Request<T>, path_username: Option<&str>) -> tide::Result<()> {
    match (req.session().get::<String>("username"), path_username) {
        (Some(sess_username), Some(path_username)) => {
            if sess_username != path_username {
                bail!(
                    401,
                    "authenticated user does not have access to requested user's data",
                )
            }
            Ok(())
        }
        (Some(_sess_username), None) => Ok(()),
        (None, _) => bail!(401, "request is not authenticated"),
    }
}

fn sanitize_urls<'s>(urls: impl Iterator<Item = &'s mut String>) -> Vec<[String; 2]> {
    let mut ret = vec![];
    for u in urls {
        if let Some(i) = u.find('?') {
            let orig = u.clone();
            u.truncate(i);
            *u = u.trim().to_string();
            ret.push([orig, u.clone()]);
        }
    }
    ret
}

async fn put_subscriptions(mut req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;
    assert_auth(&req, Some(path_username))?;
    let path_deviceid = req.param("deviceid")?;
    let (path_deviceid, format) = split_suffix(path_deviceid);
    assert_format(format)?;
    let path_deviceid = path_deviceid.to_string();
    let mut subs: Vec<Podcast> = req.body_json().await?;
    sanitize_urls(subs.iter_mut().map(|s| &mut s.url));
    let path_username = req.param("username")?;
    userdata(path_username, |userdata| {
        userdata.devsubs.push(DevSubDiff {
            deviceid: path_deviceid,
            diff: SubDiff {
                add: subs.iter().map(|p| p.url.clone()).collect(),
                remove: userdata.podcasts.iter().map(|p| p.url.clone()).collect(),
                timestamp: now(),
            },
        });
        userdata.podcasts = subs;
        Ok(())
    })?;
    Ok("".into())
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
struct SubDiff {
    #[serde(default)]
    add: Vec<String>,
    #[serde(default)]
    remove: Vec<String>,
    #[serde(default)]
    timestamp: u64,
}

#[derive(Clone, PartialEq, Eq, Deserialize, Serialize)]
struct DevSubDiff {
    deviceid: String,
    #[serde(flatten)]
    diff: SubDiff,
}

impl PartialOrd for DevSubDiff {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DevSubDiff {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.diff.timestamp,
            &self.deviceid,
            &self.diff.add,
            &self.diff.remove,
        )
            .cmp(&(
                other.diff.timestamp,
                &other.deviceid,
                &other.diff.add,
                &other.diff.remove,
            ))
    }
}

#[derive(Serialize)]
struct UpdateUrls {
    timestamp: u64,
    update_urls: Vec<[String; 2]>,
}

async fn update_subscriptions(mut req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;
    let path_deviceid = req.param("deviceid")?;
    assert_auth(&req, Some(path_username))?;
    let (path_deviceid, format) = split_suffix(path_deviceid);
    assert_format(format)?;
    let path_deviceid = path_deviceid.to_string();
    let mut diff: SubDiff = req.body_json().await?;
    log::debug!("body: {diff:?}");
    let surls = sanitize_urls(diff.add.iter_mut().chain(diff.remove.iter_mut()));
    if diff.add.is_empty() && diff.remove.is_empty() {
        return Ok("".into());
    }
    // if diff.add.iter().any(|u| diff.remove.contains(u)) {}
    let path_username = req.param("username")?;
    userdata(path_username, |userdata| {
        let mut curdiff = DevSubDiff {
            deviceid: path_deviceid,
            diff: diff.clone(),
        };
        curdiff.diff.timestamp = now();
        userdata.devsubs.push(curdiff);
        userdata.podcasts.extend(diff.add.iter().map(|url| Podcast {
            url: url.clone(),
            ..default()
        }));
        for url in diff.remove {
            if let Some((i, _)) = userdata
                .podcasts
                .iter()
                .enumerate()
                .find(|(_, p)| p.url == url)
            {
                userdata.podcasts.swap_remove(i);
            }
        }
        Ok(())
    })?;

    let body_json = serde_json::to_string(&UpdateUrls {
        timestamp: now(),
        update_urls: surls,
    })?;
    Ok(Response::builder(200)
        .body(body_json)
        .content_type("application/json")
        .build())
}

async fn get_sub_changes(req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;
    assert_auth(&req, Some(path_username))?;
    let path_deviceid = req.param("deviceid")?;
    let (path_deviceid, format) = split_suffix(path_deviceid);
    assert_format(format)?;

    let mut since = 0;
    for (k, v) in req.url().query_pairs() {
        if k.as_ref() == "since" {
            let epoch_secs = v.parse::<u64>()?;
            since = epoch_secs;
        }
    }
    let body_json = userdata(path_username, |userdata| {
        let subdiff = userdata
            .devsubs
            .iter()
            .filter(|devsub| devsub.diff.timestamp > since && devsub.deviceid == path_deviceid)
            .map(|devsub| &devsub.diff)
            .fold(
                SubDiff {
                    timestamp: now(),
                    add: vec![],
                    remove: vec![],
                },
                |mut acc, diff| {
                    acc.add.extend(diff.add.iter().cloned());
                    acc.remove.extend(diff.remove.iter().cloned());
                    acc
                },
            );
        Ok(serde_json::to_string(&subdiff)?)
    })?;
    Ok(Response::builder(200)
        .body(body_json)
        .content_type("application/json")
        .build())
}

// Events

#[derive(Default, Serialize)]
struct EpisodeActions {
    actions: Vec<Action>,
    timestamp: u64,
}

async fn get_events(req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;
    let (path_username, format) = split_suffix(path_username);
    assert_format(format)?;
    assert_auth(&req, Some(path_username))?;
    let mut since = 0;
    let mut podcast = String::new();
    let mut _aggregated = true;
    for (k, v) in req.url().query_pairs() {
        match k.as_ref() {
            "since" => since = v.parse::<u64>()?,
            "podcast" => podcast = v.to_string(),
            "aggregated" => _aggregated = v.as_ref() == "true",
            _ => {}
        }
    }
    let since = Time::UNIX_EPOCH
        .checked_add(time::Duration::seconds(since as i64))
        .unwrap();

    let body_json = userdata(path_username, |userdata| {
        let evts = userdata
            .events
            .iter()
            .filter(|evt| podcast.is_empty() || evt.podcast == podcast)
            .filter(|evt| evt.timestamp >= since);
        let epacts = evts.fold(EpisodeActions::default(), |mut acc, evt| {
            acc.actions.push(evt.action);
            acc.timestamp = (evt.timestamp - Time::UNIX_EPOCH).whole_seconds() as u64;
            acc
        });

        Ok(serde_json::to_string(&epacts)?)
    })?;

    Ok(Response::builder(200)
        .body(body_json)
        .content_type("application/json")
        .build())
}

async fn post_events(mut req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;
    let (path_username, format) = split_suffix(path_username);
    assert_format(format)?;
    assert_auth(&req, Some(path_username))?;
    let mut evts: Vec<Event> = match req.body_json().await {
        Ok(evts) => evts,
        Err(err) => {
            log::debug!("unable to deserialize: {}", req.body_string().await?);
            Err(err)?
        }
    };
    let surls = sanitize_urls(evts.iter_mut().map(|evt| &mut evt.podcast));
    let path_username = req.param("username")?;
    let (path_username, _) = split_suffix(path_username);
    userdata(path_username, |userdata| {
        userdata.events.extend(evts);
        userdata.events.sort();
        Ok(())
    })?;

    let body_json = serde_json::to_string(&UpdateUrls {
        timestamp: now(),
        update_urls: surls,
    })?;
    Ok(Response::builder(200)
        .body(body_json)
        .content_type("application/json")
        .build())
}

#[allow(unused)]
struct DebugPrintMiddleware;

#[tide::utils::async_trait]
impl<State: Clone + Send + Sync + 'static> tide::Middleware<State> for DebugPrintMiddleware {
    async fn handle(
        &self,
        mut request: Request<State>,
        next: tide::Next<'_, State>,
    ) -> tide::Result {
        // if log::log_enabled!(log::Level::Debug) {
        let req_body = request.body_string().await?;
        let res = next.run(request).await;
        log::info!("request body: {req_body}");
        Ok(res)
        // } else {
        //     Ok(next.run(request).await)
        // }
    }
}

#[async_std::main]
async fn main() {
    femme::start();

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("serve") | None => cmd_serve().await,
        Some("migrate-user") => cmd_migrate_user(&args.next().unwrap()).await,
        Some("-h") | Some("--help") => cmd_help().await,
        Some(arg0) => {
            eprintln!("unrecognized cmd {arg0:?}");
            cmd_help().await
        }
    }
}

async fn cmd_serve() {
    let mut app = tide::new();
    app.with(tide::log::LogMiddleware::new());
    app.with(tide::sessions::SessionMiddleware::new(
        tide::sessions::MemoryStore::new(),
        std::env::var("GPODRS_SESSION_SECRET")
            .expect("GPODRS_SESSION_SECRET must be set")
            .as_bytes(),
    ));
    // app.with(DebugPrintMiddleware);
    // https://gpoddernet.readthedocs.io/en/latest/api/reference/clientconfig.html
    app.at("/clientconfig.json").get(todo);
    // https://gpoddernet.readthedocs.io/en/latest/api/reference/auth.html
    app.at("/api/2/auth/:username/login.json").post(auth_login);
    app.at("/api/2/auth/:username/logout.json")
        .post(auth_logout);
    // https://gpoddernet.readthedocs.io/en/latest/api/reference/devices.html
    app.at("/api/2/devices/:username/:deviceid")
        .post(update_device);
    app.at("/api/2/devices/:username").get(list_devices);
    app.at("/api/2/devices/:username/:deviceid.json").get(todo);
    // https://gpoddernet.readthedocs.io/en/latest/api/reference/subscriptions.html
    // all have format suffixes
    app.at("/subscriptions/:username/:deviceid")
        .get(get_subscriptions);
    app.at("/subscriptions/:username").get(get_subscriptions);
    app.at("/subscriptions/:username/:deviceid")
        .put(put_subscriptions);
    app.at("/api/2/subscriptions/:username/:deviceid")
        .post(update_subscriptions);
    app.at("/api/2/subscriptions/:username/:deviceid")
        .get(get_sub_changes);
    // https://gpoddernet.readthedocs.io/en/latest/api/reference/sync.html
    app.at("/api/2/sync-devices/:username.json").get(todo);
    app.at("/api/2/sync-devices/:username.json").post(todo);
    // https://gpoddernet.readthedocs.io/en/latest/api/reference/events.html
    app.at("/api/2/episodes/:username").post(post_events);
    app.at("/api/2/episodes/:username").get(get_events);

    let listen_addr = std::env::var("GPODRS_ADDR");
    let listen_addr = listen_addr
        .as_ref()
        .map(|s| s.as_str())
        .unwrap_or("localhost:3005");
    app.listen(listen_addr).await.expect("listen");
}

async fn cmd_migrate_user(username: &str) {
    let mut cfg_v1 = std::env::var("GPODRS_CONFIG_DIR").unwrap_or("./".to_string());
    if !cfg_v1.ends_with('/') {
        cfg_v1.push('/');
    }
    cfg_v1.push_str(username);
    cfg_v1.push_str(".json");
    if !std::fs::exists(&cfg_v1).unwrap_or(false) {
        eprintln!("userdata v1 file {cfg_v1:?} not found");
        std::process::exit(1);
    }

    let datafile = std::fs::File::options()
        .read(true)
        .write(true)
        .create(false)
        .open(&cfg_v1)
        .expect("open userdata v1 file");
    let mut userdata: UserData = serde_json::from_reader(&datafile).expect("read userdata");

    let cfg_v2 = std::env::var("GPODRS_CONFIG_DIR").unwrap_or("./".to_string());
    let mut cfg_v2 = std::path::PathBuf::from(cfg_v2);
    cfg_v2.push(format!("{username}_cfg.json"));

    if std::fs::exists(&cfg_v2).unwrap_or(false) {
        eprintln!("userdata v2 file {:?} exists", cfg_v2.display());
        std::process::exit(1);
    }

    let user_cfg = std::fs::File::options()
        .read(false)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&cfg_v2)
        .expect("create userdata v2 file");
    cfg_v2.pop();
    cfg_v2.push(format!("{username}_events.json"));
    let user_events = std::fs::File::options()
        .read(false)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&cfg_v2)
        .expect("create userdata v2 events file");

    let events = std::mem::take(&mut userdata.events);

    serde_json::to_writer_pretty(user_cfg, &userdata).expect("write userdata cfg");
    for event in events {
        serde_json::to_writer(&user_events, &event).expect("write userdata event");
        writeln!(&user_events).unwrap();
    }

    println!("migrated {username} config to v2");
}

async fn cmd_help() {
    eprintln!("usage: gpodrs [-h] [CMD]");
    eprintln!("CMD:");
    eprintln!("    serve         # default");
    eprintln!("    migrate-user");
}

#[cfg(test)]
mod test {
    use std::path::Path;

    use super::*;

    #[test]
    fn test_minus_opt() {
        let act: Result<Action, _> = serde_json::from_str(
            r#"{"action": "play", "started": -1, "position": -1, "total": -1}"#,
        )
        .map_err(|err| err.to_string());
        assert_eq!(
            Ok(Action::Play {
                started: None,
                position: None,
                total: None
            }),
            act
        );
        let act: Result<Action, _> = serde_json::from_str(
            r#"{"action": "play", "started": null, "position": null, "total": null}"#,
        )
        .map_err(|err| err.to_string());
        assert_eq!(
            Ok(Action::Play {
                started: None,
                position: None,
                total: None
            }),
            act
        );
    }

    #[test]
    fn test_event_timestamp() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Example {
            #[serde(with = "timestamp")]
            time: Time,
        }

        let time = Time::from_unix_timestamp(1753911230).unwrap();
        assert_eq!(
            serde_json::to_string(&Example { time }).unwrap(),
            r#"{"time":"2025-07-30T21:33:50"}"#
        );

        assert_eq!(
            Example { time },
            serde_json::from_str(r#"{"time":"2025-07-30T21:33:50"}"#).unwrap(),
        );
    }

    #[test]
    fn test_update_events() {
        femme::with_level(log::LevelFilter::Debug);

        #[track_caller]
        fn assert_file(path: impl AsRef<Path>, contents: &str) {
            let actual = std::fs::read_to_string(path.as_ref()).expect("file exists");
            assert!(
                actual == contents,
                "file contents don't match expected\n\nACTUAL: {path}\n{actual}\n\nEXPECTED:\n{contents}",
                path = path.as_ref().display(),
            );
        }

        struct Defer<F: FnOnce()>(Option<F>);
        impl<F: FnOnce()> std::ops::Drop for Defer<F> {
            fn drop(&mut self) {
                if let Some(f) = self.0.take() {
                    f()
                }
            }
        }

        let _cleanup = Defer(Some(|| {
            let _ = std::fs::remove_file("test_user2_cfg.json");
            let _ = std::fs::remove_file("test_user2_events.json");
        }));
        let test_user_cfg = std::fs::File::create("test_user2_cfg.json").unwrap();
        let data = UserData {
            username: "test_user2".to_string(),
            password: "".to_string(),
            podcasts: vec![],
            events: vec![],
            devices: vec![],
            devsubs: vec![],
        };
        serde_json::to_writer_pretty(&test_user_cfg, &data).unwrap();

        let events_path = "test_user2_events.json";
        std::fs::File::create(events_path).unwrap();

        let mut cur_time = Time::UNIX_EPOCH;
        let mut event = Event {
            podcast: "nothing".to_string(),
            episode: "episode".to_string(),
            device: "device".to_string(),
            action: Action::Play {
                started: Some(cur_time.unix_timestamp().try_into().unwrap()),
                position: Some(0),
                total: Some(0),
            },
            timestamp: cur_time,
        };

        assert_file(events_path, "");

        userdata("test_user2", |data2| {
            data2.events.push(event.clone());
            Ok(())
        })
        .unwrap();

        assert_file(
            events_path,
            r#"{"podcast":"nothing","episode":"episode","device":"device","action":"play","started":0,"position":0,"total":0,"timestamp":"1970-01-01T00:00:00"}
"#,
        );

        cur_time += std::time::Duration::from_secs(2);
        event.action = Action::Play {
            started: None,
            position: Some(2),
            total: Some(0),
        };
        event.timestamp = cur_time;

        userdata("test_user2", |data2| {
            data2.events.push(event.clone());
            Ok(())
        })
        .unwrap();

        assert_file(
            events_path,
            r#"{"podcast":"nothing","episode":"episode","device":"device","action":"play","started":0,"position":0,"total":0,"timestamp":"1970-01-01T00:00:00"}
{"podcast":"nothing","episode":"episode","device":"device","action":"play","started":null,"position":2,"total":0,"timestamp":"1970-01-01T00:00:02"}
"#,
        );

        event.action = Action::Play {
            started: None,
            position: Some(1),
            total: Some(0),
        };
        event.timestamp = cur_time - std::time::Duration::from_secs(1);

        userdata("test_user2", |data2| {
            data2.events.push(event.clone());
            Ok(())
        })
        .unwrap();

        assert_file(
            events_path,
            r#"{"podcast":"nothing","episode":"episode","device":"device","action":"play","started":0,"position":0,"total":0,"timestamp":"1970-01-01T00:00:00"}
{"podcast":"nothing","episode":"episode","device":"device","action":"play","started":null,"position":2,"total":0,"timestamp":"1970-01-01T00:00:02"}
{"podcast":"nothing","episode":"episode","device":"device","action":"play","started":null,"position":1,"total":0,"timestamp":"1970-01-01T00:00:01"}
"#,
        );
    }
}
