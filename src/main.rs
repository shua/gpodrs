use anyhow::anyhow;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use log;
use serde::{Deserialize, Serialize};
use serde_json;
use std::time::{Duration, SystemTime as Time};
use tide::{Request, Response};

// https://github.com/bohwaz/micro-gpodder-server
// https://github.com/ahgamut/rust-ape-example

fn default<T: std::default::Default>() -> T {
    T::default()
}

fn now() -> u64 {
    Time::now()
        .duration_since(Time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

mod timestamp {
    use std::ffi::{c_char, c_long};
    use std::time::{Duration, SystemTime as Time};

    const ISO_UTC_FMT: *const i8 = "%FT%T\0".as_ptr().cast();

    #[repr(C)]
    #[allow(non_camel_case_types)]
    struct tm {
        tm_sec: i32,
        tm_min: i32,
        tm_hour: i32,
        tm_mday: i32,
        tm_mon: i32,
        tm_year: i32,
        tm_wday: i32,
        tm_yday: i32,
        tm_isdst: i32,

        tm_gmtoff: i64,
        tm_zone: *const c_char,
    }

    extern "C" {
        static timezone: c_long;
        fn tzset();
        fn strftime(buf: *mut c_char, max: isize, format: *const c_char, tm: *const tm) -> isize;
        fn strptime(s: *const c_char, format: *const c_char, tm: *mut tm) -> *const c_char;
        fn gmtime(timep: *const c_long) -> *mut tm;
        fn mktime(tm: *mut tm) -> c_long;
    }

    pub fn init() {
        std::env::set_var("TZ", "UTC");
        unsafe { tzset() };
        assert_eq!(
            unsafe { timezone },
            0,
            "timezone should be set to 0 seconds from UTC"
        );
    }

    pub fn serialize<S: serde::Serializer>(t: &Time, s: S) -> Result<S::Ok, S::Error> {
        let epoch_time = t.duration_since(Time::UNIX_EPOCH).unwrap();
        let epoch_secs = epoch_time.as_secs();
        let mut ret = String::from("1994-05-06T07:08:09Z_");
        unsafe {
            let tm = gmtime(&(epoch_secs as i64));
            let len = strftime(
                ret.as_mut_ptr() as *mut i8,
                ret.capacity() as isize,
                ISO_UTC_FMT,
                tm,
            );
            ret.truncate(len as usize);
        }
        s.serialize_str(&ret)
    }

    struct TimestampVisitor;

    impl<'de> serde::de::Visitor<'de> for TimestampVisitor {
        type Value = Time;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("ISO timestamp")
        }
        fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<Self::Value, E> {
            if s.is_empty() {
                return Ok(Time::UNIX_EPOCH);
            }

            let c_s = format!("{s}\0");
            let c_s_len = c_s.len();
            let c_s: *const i8 = c_s.as_ptr().cast();
            unsafe {
                let mut tm: tm = std::mem::zeroed();
                let ret = strptime(c_s, ISO_UTC_FMT, &mut tm);
                if ret.is_null() || ret.offset_from(c_s) + 1 != c_s_len as isize {
                    log::error!(
                        "strptime({:?}, ISO_FMT, ...) = {:?} doesn't match expected {:?}+{:x}",
                        c_s,
                        ret,
                        c_s,
                        c_s_len
                    );
                    return Err(E::invalid_value(serde::de::Unexpected::Str(s), &self));
                }
                let epoch_secs = mktime(&mut tm);
                if epoch_secs == -1 {
                    log::error!("error calling mktime");
                    return Err(E::invalid_value(serde::de::Unexpected::Str(s), &self));
                }
                Ok(Time::UNIX_EPOCH
                    .checked_add(Duration::from_secs(epoch_secs as u64))
                    .unwrap())
            }
        }
        fn visit_u64<E: serde::de::Error>(self, n: u64) -> Result<Self::Value, E> {
            Ok(Time::UNIX_EPOCH
                .checked_add(Duration::from_secs(n))
                .unwrap())
        }
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(dsr: D) -> Result<Time, D::Error> {
        dsr.deserialize_str(TimestampVisitor)
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "lowercase")]
enum Action {
    Add,
    Remove,
    Download,
    Delete,
    Play {
        started: Option<u64>,
        position: Option<u64>,
        total: Option<u64>,
    },
    New,
    Flattr,
}

#[derive(Clone, PartialEq, Eq, Ord, Serialize, Deserialize)]
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
        (
            self.timestamp,
            &self.podcast,
            &self.episode,
            &self.action,
            &self.device,
        )
            .partial_cmp(&(
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
    events: Vec<Event>,
    devices: Vec<Device>,
    devsubs: Vec<DevSubDiff>,
}

fn userdata<R>(f: impl FnOnce(&mut UserData) -> tide::Result<R>) -> tide::Result<R> {
    use std::io::{Seek, SeekFrom};
    let mut cfg = std::env::var("GPODRS_CONFIG_DIR").unwrap_or("./".to_string());
    if !cfg.ends_with('/') {
        cfg.push('/');
    }
    cfg.push_str("shua.json");
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

// auth

async fn auth_login(mut req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;

    if let Some(sess_username) = req.session().get::<String>("username") {
        // check sessionid
        if sess_username != path_username {
            return Err(tide::Error::new(
                400,
                anyhow!("session username is not valid for authenticated user"),
            ));
        }
        return Ok("".into());
    }

    // normal auth flow
    let auth_hdr = req
        .header("Authorization")
        .map(|hdrs| hdrs.last())
        .ok_or(tide::Error::new(
            401,
            anyhow!("authorization header not present"),
        ))?;
    if !auth_hdr.as_str().starts_with("Basic ") {
        return Err(tide::Error::new(
            401,
            anyhow!("authorize header is not Basic"),
        ));
    }
    let auth_hdr = BASE64
        .decode(&auth_hdr.as_str()["Basic ".len()..])
        .map_err(|e| tide::Error::new(401, e))?;
    let auth_hdr = String::from_utf8(auth_hdr).map_err(|e| tide::Error::new(401, e))?;
    let (auth_username, auth_password) = (auth_hdr
        .find(":")
        .map(|i| auth_hdr.split_at(i))
        .map(|(u, p)| (u, &p[1..])))
    .ok_or(tide::Error::new(
        401,
        anyhow!("authorize header is not valid basic auth"),
    ))?;
    if auth_username != path_username {
        return Err(tide::Error::new(
            401,
            anyhow!("login username does not match path resource"),
        ));
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
    match userdata(|userdata| {
        Ok(path_username == ""
            || userdata.username != auth_username
            || userdata.password != auth_password_hash)
    }) {
        Err(err) => {
            if let Some(ioerr) = err.downcast_ref::<std::io::Error>() {
                if ioerr.kind() == std::io::ErrorKind::NotFound {
                    log::error!("{ioerr}");
                    return Err(tide::Error::new(
                        401,
                        anyhow!("user {auth_username} does not exist"),
                    ));
                }
            }
            return Err(err);
        }
        Ok(false) => {}
        Ok(true) => {
            return Err(tide::Error::new(
                401,
                anyhow!("unable to authenticate: {auth_username:?} {auth_password:?}"),
            ));
        }
    }

    req.session_mut().insert("username", auth_username)?;
    Ok("".into())
}

async fn auth_logout(mut req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;
    if let Some(sess_username) = req.session().get::<String>("username") {
        if path_username != sess_username {
            return Err(tide::Error::new(
                400,
                anyhow!("session user does not match path resource"),
            ));
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
        Err(tide::Error::new(400, anyhow!("no format specified")))
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
    let body_json = userdata(|userdata| Ok(serde_json::to_string(&userdata.devices)?))?;
    Ok(Response::builder(200)
        .body(body_json)
        .content_type("application/json")
        .build())
}

async fn update_device(mut req: Request<()>) -> tide::Result {
    let path_username = req.param("username")?;
    assert_auth(&req, Some(path_username))?;
    let path_deviceid = req.param("deviceid")?;
    let (path_deviceid, format) = split_suffix(&path_deviceid);
    let path_deviceid = path_deviceid.to_string();
    assert_format(format)?;

    let mut device: Device = req.body_json().await?;
    device.id = path_deviceid;
    if device.id != "" {
        userdata(|userdata| {
            if let Some(dev) = userdata.devices.iter_mut().find(|d| d.id == device.id) {
                if device.caption != "" {
                    dev.caption = device.caption;
                }
                if device.r#type != "" {
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

    let subs = userdata(|userdata| Ok(serde_json::to_string(&userdata.podcasts)?))?;
    Ok(Response::builder(200)
        .content_type("application/json")
        .body(subs)
        .build())
}

fn assert_auth<T>(req: &Request<T>, path_username: Option<&str>) -> tide::Result<()> {
    match (req.session().get::<String>("username"), path_username) {
        (Some(sess_username), Some(path_username)) => {
            if sess_username != path_username {
                Err(tide::Error::new(
                    401,
                    anyhow!("authenticated user does not have access to requested user's data"),
                ))
            } else {
                Ok(())
            }
        }
        (Some(_sess_username), None) => Ok(()),
        (None, _) => Err(tide::Error::new(
            401,
            anyhow!("request is not authenticated"),
        )),
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
    userdata(|userdata| {
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
        (
            self.diff.timestamp,
            &self.deviceid,
            &self.diff.add,
            &self.diff.remove,
        )
            .partial_cmp(&(
                other.diff.timestamp,
                &other.deviceid,
                &other.diff.add,
                &other.diff.remove,
            ))
    }
}

impl Ord for DevSubDiff {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
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
    println!("body: {diff:?}");
    let surls = sanitize_urls(diff.add.iter_mut().chain(diff.remove.iter_mut()));
    if diff.add.len() == 0 && diff.remove.len() == 0 {
        return Ok("".into());
    }
    if diff.add.iter().any(|u| diff.remove.contains(u)) {}
    userdata(|userdata| {
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
            if let Some((i, _)) = (userdata.podcasts.iter().enumerate())
                .filter(|(_, p)| p.url == url)
                .next()
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
        match k.as_ref() {
            "since" => {
                let epoch_secs = u64::from_str_radix(&v, 10)?;
                since = epoch_secs;
            }
            _ => {}
        }
    }
    let body_json = userdata(|userdata| {
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
            "since" => since = u64::from_str_radix(&v, 10)?,
            "podcast" => podcast = v.to_string(),
            "aggregated" => _aggregated = v.as_ref() == "true",
            _ => {}
        }
    }
    let since = Time::UNIX_EPOCH
        .checked_add(Duration::from_secs(since))
        .unwrap();

    let body_json = userdata(|userdata| {
        let evts = userdata
            .events
            .iter()
            .filter(|evt| podcast == "" || evt.podcast == podcast)
            .filter(|evt| evt.timestamp >= since);
        let epacts = evts.fold(EpisodeActions::default(), |mut acc, evt| {
            acc.actions.push(evt.action.clone());
            acc.timestamp = evt
                .timestamp
                .duration_since(Time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
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
    let mut evts: Vec<Event> = req.body_json().await?;
    let surls = sanitize_urls(evts.iter_mut().map(|evt| &mut evt.podcast));
    userdata(|userdata| {
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

#[async_std::main]
async fn main() {
    timestamp::init();
    femme::start();
    let mut app = tide::new();
    app.with(tide::log::LogMiddleware::new());
    app.with(tide::sessions::SessionMiddleware::new(
        tide::sessions::MemoryStore::new(),
        std::env::var("GPODRS_SESSION_SECRET")
            .expect("GPODRS_SESSION_SECRET must be set")
            .as_bytes(),
    ));
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
