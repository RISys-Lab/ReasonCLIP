#!/bin/bash

# Released model list for eval scripts.
# All ReasonCLIP/ReasonSigLIP release repos include their own processor files.
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
  "RISys-Lab/ReasonSigLIP-So14-384-S0-Rea"
  "RISys-Lab/ReasonSigLIP-So14-384-S0-Des"
  "RISys-Lab/ReasonSigLIP-So14-384-S1"
  "RISys-Lab/ReasonSigLIP-So14-384-S2"

  "google/siglip2-so400m-patch14-384"
  "RISys-Lab/ReasonSigLIP2-So14-384-S0-Rea"
  "RISys-Lab/ReasonSigLIP2-So14-384-S0-Des"
  "RISys-Lab/ReasonSigLIP2-So14-384-S1"
  "RISys-Lab/ReasonSigLIP2-So14-384-S2"

  "google/siglip2-giant-opt-patch16-384"
  "RISys-Lab/ReasonSigLIP2-go16-384-S0-Rea"
  "RISys-Lab/ReasonSigLIP2-go16-384-S0-Des"
  "RISys-Lab/ReasonSigLIP2-go16-384-S1"
  "RISys-Lab/ReasonSigLIP2-go16-384-S2"
)
