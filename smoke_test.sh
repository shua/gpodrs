#!/bin/sh


api() {
  printf '+ api %s\n' "$*"
  path=$1
  shift
  curl -b cookies -c cookies localhost:3005/api/2/$path "$@"
  echo
}

api auth/test_user/login.json -u test_user:test_password -X POST
api devices/test_user.json
api subscriptions/test_user.json
api subscriptions/test_user/jelly2.json
api episodes/test_user.json
