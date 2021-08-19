#! /bin/sh
curl \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"access_key":"'"$BACKENDAI_ACCESS_KEY"'","image":"'"$BACKENDAI_KERNEL_IMAGE"'"}' \
  https://webhook.cloud.backend.ai/3rdparties/aifactory