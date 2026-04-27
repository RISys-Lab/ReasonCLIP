#!/bin/bash

# Shared model/processor lists for eval scripts.
# Keep these arrays aligned by index.
models=(
  "fesvhtr/clip-r-336-rea-run1219-608"
  "fesvhtr/clip-r-336-des-run0201-949"
  "fesvhtr/clip-r-336-s1-run1215-1280"
  "fesvhtr/clip-r-336-s2-run0204-505"

  "fesvhtr/clip-r-b32-rea-run1219-466"
  "fesvhtr/clip-r-b32-des-run0201-949"
  "fesvhtr/clip-r-b32-s1-run0109-853"
  "fesvhtr/clip-r-b32-s2-run0205-336"
  "fesvhtr/clip-r-b32-s1-read"

  "fesvhtr/clip-r-rea-run1219-621"
  "fesvhtr/clip-r-des-run0131-949"
  "fesvhtr/clip-r-s1-run1207-1280"
  "fesvhtr/clip-r-s2-run0204-505"


  "fesvhtr/siglip-r-rea-run0126-1241"
  "fesvhtr/siglip-r-des-run0131-1266"
  "fesvhtr/siglip-r-s1-run0201-1280"
  "fesvhtr/siglip-r-s2-run0203-673"
  
  "fesvhtr/siglip2-r-des-run0206-949"
  "fesvhtr/siglip2-r-rea-run0208-931"
  "fesvhtr/siglip2-r-s1-run0205-1280"
  "fesvhtr/siglip2-r-s2-run0206-505"

  "fesvhtr/siglip2-r-go-des-run0206-1266"
  "fesvhtr/siglip2-r-go-rea-run0207-1241"
  "fesvhtr/siglip2-r-go-s1-run0205-1706"
  "fesvhtr/siglip2-r-go-s2-run0206-673"
)

processors=(
  "openai/clip-vit-large-patch14-336"
  "openai/clip-vit-large-patch14-336"
  "openai/clip-vit-large-patch14-336"
  "openai/clip-vit-large-patch14-336"

  "openai/clip-vit-base-patch32"
  "openai/clip-vit-base-patch32"
  "openai/clip-vit-base-patch32"
  "openai/clip-vit-base-patch32"
  "openai/clip-vit-base-patch32"

  "openai/clip-vit-large-patch14"
  "openai/clip-vit-large-patch14"
  "openai/clip-vit-large-patch14"
  "openai/clip-vit-large-patch14"

  "google/siglip-so400m-patch14-384"
  "google/siglip-so400m-patch14-384"
  "google/siglip-so400m-patch14-384"
  "google/siglip-so400m-patch14-384"

  "google/siglip2-so400m-patch14-384"
  "google/siglip2-so400m-patch14-384"
  "google/siglip2-so400m-patch14-384"
  "google/siglip2-so400m-patch14-384"

  "google/siglip2-giant-opt-patch16-384"
  "google/siglip2-giant-opt-patch16-384"
  "google/siglip2-giant-opt-patch16-384"
  "google/siglip2-giant-opt-patch16-384"
)
