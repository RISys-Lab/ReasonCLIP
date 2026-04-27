#!/bin/bash

# Shared model/processor lists for eval scripts.
# Keep these arrays aligned by index.
models=(
  "openai/clip-vit-large-patch14-336"
  "RISys-Lab/ReasonCLIP-L14-336-S0-Rea"
  "RISys-Lab/ReasonCLIP-L14-336-S0-Des"
  "RISys-Lab/ReasonCLIP-L14-336-S1"
  "RISys-Lab/ReasonCLIP-L14-336-S2"

  "openai/clip-vit-base-patch32"
  "RISys-Lab/ReasonCLIP-B32-S0-Rea"
  "RISys-Lab/ReasonCLIP-B32-S0-Des"
  "RISys-Lab/ReasonCLIP-B32-S1"
  "RISys-Lab/ReasonCLIP-B32-S2"
  "RISys-Lab/ReasonCLIP-B32-READ"

  "openai/clip-vit-large-patch14"
  "RISys-Lab/ReasonCLIP-L14-224-S0-Rea"
  "RISys-Lab/ReasonCLIP-L14-224-S0-Des"
  "RISys-Lab/ReasonCLIP-L14-224-S1"
  "RISys-Lab/ReasonCLIP-L14-224-S2"

  "google/siglip-so400m-patch14-384"
  "RISys-Lab/ReasonSigLIP-So14-S0-Rea"
  "RISys-Lab/ReasonSigLIP-So14-S0-Des"
  "RISys-Lab/ReasonSigLIP-So14-S1"
  "RISys-Lab/ReasonSigLIP-So14-S2"

  "google/siglip2-so400m-patch14-384"
  "RISys-Lab/ReasonSigLIP2-So14-S0-Rea"
  "RISys-Lab/ReasonSigLIP2-So14-S0-Des"
  "RISys-Lab/ReasonSigLIP2-So14-S1"
  "RISys-Lab/ReasonSigLIP2-So14-S2"

  "google/siglip2-giant-opt-patch16-384"
  "RISys-Lab/ReasonSigLIP2-GO16-S0-Rea"
  "RISys-Lab/ReasonSigLIP2-GO16-S0-Des"
  "RISys-Lab/ReasonSigLIP2-GO16-S1"
  "RISys-Lab/ReasonSigLIP2-GO16-S2"
)

processors=(
  "openai/clip-vit-large-patch14-336"
  "openai/clip-vit-large-patch14-336"
  "openai/clip-vit-large-patch14-336"
  "openai/clip-vit-large-patch14-336"
  "openai/clip-vit-large-patch14-336"

  "openai/clip-vit-base-patch32"
  "openai/clip-vit-base-patch32"
  "openai/clip-vit-base-patch32"
  "openai/clip-vit-base-patch32"
  "openai/clip-vit-base-patch32"
  "openai/clip-vit-base-patch32"

  "openai/clip-vit-large-patch14"
  "openai/clip-vit-large-patch14"
  "openai/clip-vit-large-patch14"
  "openai/clip-vit-large-patch14"
  "openai/clip-vit-large-patch14"

  "google/siglip-so400m-patch14-384"
  "google/siglip-so400m-patch14-384"
  "google/siglip-so400m-patch14-384"
  "google/siglip-so400m-patch14-384"
  "google/siglip-so400m-patch14-384"

  "google/siglip2-so400m-patch14-384"
  "google/siglip2-so400m-patch14-384"
  "google/siglip2-so400m-patch14-384"
  "google/siglip2-so400m-patch14-384"
  "google/siglip2-so400m-patch14-384"

  "google/siglip2-giant-opt-patch16-384"
  "google/siglip2-giant-opt-patch16-384"
  "google/siglip2-giant-opt-patch16-384"
  "google/siglip2-giant-opt-patch16-384"
  "google/siglip2-giant-opt-patch16-384"
)
