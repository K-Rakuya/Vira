#!/usr/bin/env bash
# ==============================================================================
# Vira Offline Voice Pipeline - Model Asset Provisioning Script
# ==============================================================================
# Downloads and verifies production-ready zero-VRAM CPU models for:
# - Piper VITS Vietnamese TTS (vi_VN-vais1000-medium)
# - Kokoro v1.0 Multilingual TTS (kokoro-multi-lang-v1_0)
# - Sherpa-ONNX Streaming Zipformer Vietnamese STT (sherpa-onnx-zipformer-vi-30M-int8)
# - Silero VAD v5 ONNX (silero_vad.onnx)
# ==============================================================================

set -euo pipefail

# ANSI Color Codes
if [[ -t 1 ]]; then
  BOLD=$'\e[1m'
  GREEN=$'\e[0;32m'
  YELLOW=$'\e[0;33m'
  BLUE=$'\e[0;34m'
  RED=$'\e[0;31m'
  CYAN=$'\e[0;36m'
  RESET=$'\e[0m'
else
  BOLD=""
  GREEN=""
  YELLOW=""
  BLUE=""
  RED=""
  CYAN=""
  RESET=""
fi

# Locate Repository Root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default storage directory strictly inside the repository (honors $VIRA_MODELS_DIR)
DEFAULT_DIR="${VIRA_MODELS_DIR:-${REPO_ROOT}/models}"
MODELS_DIR="$DEFAULT_DIR"

# Flags
DO_TTS_VI=false
DO_TTS_EN=false
DO_STT_VI=false
DO_STT_EN=false
DO_VAD=false
DO_CHECK=false

# ------------------------------------------------------------------------------
# Model Metadata (URLs, SHA256 checksums, and directory targets)
# ------------------------------------------------------------------------------

# 1. Silero VAD v5
VAD_FILENAME="silero_vad.onnx"
VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
VAD_BACKUP_URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
VAD_SHA256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"

# 2. Piper Vietnamese TTS
TTS_VI_DIR_NAME="vits-piper-vi_VN-vais1000-medium"
TTS_VI_ARCHIVE="vits-piper-vi_VN-vais1000-medium.tar.bz2"
TTS_VI_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-vi_VN-vais1000-medium.tar.bz2"
TTS_VI_SHA256="fa1367710767d36ed5cf13b4a449e20c35ffd12791c2e47c2e64142bfa55551a"

# 3. Kokoro Multilingual (EN/JA) TTS
TTS_EN_DIR_NAME="kokoro-multi-lang-v1_0"
TTS_EN_ARCHIVE="kokoro-multi-lang-v1_0.tar.bz2"
TTS_EN_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
TTS_EN_SHA256="60703707bc05ccd48d60e1eb723f38b07d6e474795a0b6a588d7220400e0f7a4"

# 4. Sherpa-ONNX Zipformer Vietnamese STT
STT_VI_DIR_NAME="sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
STT_VI_ARCHIVE="sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"
STT_VI_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"
STT_VI_SHA256="da8b637947091829d7ee9eda23da2a4ec7caa399233a3f4e34eb719fb2ea6b9b"

# 5. Sherpa-ONNX Streaming Zipformer English STT (Optional)
STT_EN_DIR_NAME="sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
STT_EN_ARCHIVE="sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
STT_EN_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

print_usage() {
  cat << EOF
${BOLD}Vira Offline Voice Pipeline Model Downloader${RESET}

Usage: $(basename "$0") [OPTIONS]

Options:
  --all           Download all voice models (TTS Vi/En, STT Vi, VAD)
  --tts-vi        Download Piper Vietnamese TTS (vi_VN-vais1000-medium) (~64 MB)
  --tts-en        Download Kokoro v1.0 Multilingual/English TTS (~333 MB)
  --stt-vi        Download Sherpa-ONNX Zipformer Vietnamese STT (~25 MB)
  --stt-en        Download Sherpa-ONNX Zipformer English STT (~122 MB, optional)
  --vad           Download Silero VAD v5 ONNX (~0.6 MB)
  --dir <PATH>    Set destination directory (Default: ./models inside repository)
  --check         Check installation and integrity of models in target directory
  -h, --help      Display this help message and exit

Environment Variables:
  VIRA_MODELS_DIR Override the default model storage directory path (defaults to ./models)

Examples:
  # Check status of models in default repository directory:
  $(basename "$0") --check

  # Download all models to ./models:
  $(basename "$0") --all

  # Download only Vietnamese TTS and STT models to custom path:
  $(basename "$0") --tts-vi --stt-vi --dir ./models
EOF
}

check_dependencies() {
  local missing=()
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    missing+=("curl or wget")
  fi
  if ! command -v tar >/dev/null 2>&1; then
    missing+=("tar")
  fi
  if ! command -v bzip2 >/dev/null 2>&1; then
    missing+=("bzip2")
  fi
  if ! command -v sha256sum >/dev/null 2>&1; then
    missing+=("sha256sum")
  fi

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo -e "${RED}Error: Missing required system utilities: ${missing[*]}${RESET}" >&2
    echo "Please install them using your package manager (e.g. apt, pacman, dnf) and retry." >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local dest="$2"
  local backup_url="${3:-}"

  local dest_tmp="${dest}.tmp.$$"

  echo -e "${CYAN}--> Downloading:${RESET} ${BOLD}${url}${RESET}"
  local success=false

  if command -v curl >/dev/null 2>&1; then
    if curl -fSL --progress-bar "$url" -o "$dest_tmp"; then
      success=true
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget --progress=bar:force -O "$dest_tmp" "$url"; then
      success=true
    fi
  fi

  if [[ "$success" == false ]] && [[ -n "$backup_url" ]]; then
    echo -e "${YELLOW}Primary URL failed. Trying mirror:${RESET} $backup_url"
    if command -v curl >/dev/null 2>&1; then
      if curl -fSL --progress-bar "$backup_url" -o "$dest_tmp"; then
        success=true
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget --progress=bar:force -O "$dest_tmp" "$backup_url"; then
        success=true
      fi
    fi
  fi

  if [[ "$success" == false ]]; then
    rm -f "$dest_tmp"
    echo -e "${RED}Error: Failed to download ${url}${RESET}" >&2
    return 1
  fi

  mv -f "$dest_tmp" "$dest"
  return 0
}

verify_sha256() {
  local file="$1"
  local expected_sha="$2"

  if [[ -z "$expected_sha" ]]; then
    return 0
  fi

  echo -ne "${CYAN}--> Verifying integrity (SHA256)... ${RESET}"
  local actual_sha
  actual_sha=$(sha256sum "$file" | awk '{print $1}')

  if [[ "$actual_sha" == "$expected_sha" ]]; then
    echo -e "${GREEN}OK${RESET}"
    return 0
  else
    echo -e "${RED}FAILED${RESET}"
    echo -e "${RED}Expected: ${expected_sha}${RESET}" >&2
    echo -e "${RED}Actual:   ${actual_sha}${RESET}" >&2
    return 1
  fi
}

extract_archive() {
  local archive="$1"
  local target_dir="$2"

  echo -e "${CYAN}--> Extracting archive:${RESET} $(basename "$archive") into ${target_dir}"
  tar -xjf "$archive" -C "$target_dir"
}

create_symlink_alias() {
  local target="$1"
  local link_name="$2"
  if [[ -e "$target" ]]; then
    ln -sfn "$(basename "$target")" "$link_name"
  fi
}

# ------------------------------------------------------------------------------
# Check Mode
# ------------------------------------------------------------------------------

check_models() {
  echo -e "${BOLD}================================================================${RESET}"
  echo -e "${BOLD}  Vira Offline Voice Pipeline — Model Status Report             ${RESET}"
  echo -e "${BOLD}================================================================${RESET}"
  echo -e "Target Directory: ${CYAN}${MODELS_DIR}${RESET}\n"

  if [[ ! -d "$MODELS_DIR" ]]; then
    echo -e "${YELLOW}Warning: Directory does not exist yet: ${MODELS_DIR}${RESET}"
    echo -e "Run ${BOLD}$(basename "$0") --all${RESET} to initialize and download models.\n"
  fi

  local all_ok=true

  # 1. Silero VAD
  local vad_file="${MODELS_DIR}/${VAD_FILENAME}"
  echo -ne "1. Silero VAD v5 (${VAD_FILENAME}): "
  if [[ -f "$vad_file" ]] && [[ -s "$vad_file" ]]; then
    local sz
    sz=$(du -h "$vad_file" | awk '{print $1}')
    echo -e "${GREEN}[INSTALLED]${RESET} (${sz})"
  else
    echo -e "${YELLOW}[MISSING]${RESET}"
    all_ok=false
  fi

  # 2. Piper Vietnamese TTS
  local tts_vi_dir="${MODELS_DIR}/${TTS_VI_DIR_NAME}"
  local tts_vi_onnx="${tts_vi_dir}/vi_VN-vais1000-medium.onnx"
  local tts_vi_tokens="${tts_vi_dir}/tokens.txt"
  echo -ne "2. Piper Vietnamese TTS (${TTS_VI_DIR_NAME}): "
  if [[ -f "$tts_vi_onnx" ]] && [[ -f "$tts_vi_tokens" ]]; then
    local sz
    sz=$(du -sh "$tts_vi_dir" 2>/dev/null | awk '{print $1}')
    echo -e "${GREEN}[INSTALLED]${RESET} (${sz})"
  else
    echo -e "${YELLOW}[MISSING]${RESET}"
    all_ok=false
  fi

  # 3. Kokoro Multilingual TTS
  local tts_en_dir="${MODELS_DIR}/${TTS_EN_DIR_NAME}"
  local tts_en_onnx1="${tts_en_dir}/model.onnx"
  local tts_en_onnx2="${tts_en_dir}/kokoro-multi-lang-v1_0.onnx"
  local tts_en_voices="${tts_en_dir}/voices.bin"
  local tts_en_tokens="${tts_en_dir}/tokens.txt"
  echo -ne "3. Kokoro Multilingual TTS (${TTS_EN_DIR_NAME}): "
  if { [[ -f "$tts_en_onnx1" ]] || [[ -f "$tts_en_onnx2" ]]; } && [[ -f "$tts_en_voices" ]] && [[ -f "$tts_en_tokens" ]]; then
    local sz
    sz=$(du -sh "$tts_en_dir" 2>/dev/null | awk '{print $1}')
    echo -e "${GREEN}[INSTALLED]${RESET} (${sz})"
  else
    echo -e "${YELLOW}[MISSING]${RESET}"
    all_ok=false
  fi

  # 4. Sherpa-ONNX Zipformer Vietnamese STT
  local stt_vi_dir="${MODELS_DIR}/${STT_VI_DIR_NAME}"
  local stt_vi_enc="${stt_vi_dir}/encoder.int8.onnx"
  local stt_vi_dec="${stt_vi_dir}/decoder.onnx"
  local stt_vi_join="${stt_vi_dir}/joiner.int8.onnx"
  local stt_vi_tokens="${stt_vi_dir}/tokens.txt"
  echo -ne "4. Zipformer Vietnamese STT (${STT_VI_DIR_NAME}): "
  if [[ -f "$stt_vi_enc" ]] && [[ -f "$stt_vi_dec" ]] && [[ -f "$stt_vi_join" ]] && [[ -f "$stt_vi_tokens" ]]; then
    local sz
    sz=$(du -sh "$stt_vi_dir" 2>/dev/null | awk '{print $1}')
    echo -e "${GREEN}[INSTALLED]${RESET} (${sz})"
  else
    echo -e "${YELLOW}[MISSING]${RESET}"
    all_ok=false
  fi

  # 5. Sherpa-ONNX Zipformer English STT (Optional)
  local stt_en_dir="${MODELS_DIR}/${STT_EN_DIR_NAME}"
  local stt_en_tokens="${stt_en_dir}/tokens.txt"
  echo -ne "5. Zipformer English STT (Optional, ${STT_EN_DIR_NAME}): "
  if [[ -d "$stt_en_dir" ]] && [[ -f "$stt_en_tokens" ]]; then
    local sz
    sz=$(du -sh "$stt_en_dir" 2>/dev/null | awk '{print $1}')
    echo -e "${GREEN}[INSTALLED]${RESET} (${sz})"
  else
    echo -e "${CYAN}[NOT INSTALLED]${RESET} (Optional)"
  fi

  echo -e "\n${BOLD}Aliases & Symlinks:${RESET}"
  local found_alias=false
  for alias in "piper" "kokoro" "streaming-zipformer" "tts-vi" "tts-en" "stt-vi" "stt-en"; do
    if [[ -L "${MODELS_DIR}/${alias}" ]]; then
      local dest
      dest=$(readlink "${MODELS_DIR}/${alias}")
      echo -e "  - ${CYAN}${alias}${RESET} -> ${dest}"
      found_alias=true
    fi
  done
  if [[ -f "${MODELS_DIR}/silero_vad/${VAD_FILENAME}" ]]; then
    echo -e "  - ${CYAN}silero_vad/${VAD_FILENAME}${RESET} -> present"
    found_alias=true
  fi
  if [[ -f "${MODELS_DIR}/vad/${VAD_FILENAME}" ]]; then
    echo -e "  - ${CYAN}vad/${VAD_FILENAME}${RESET} -> present"
    found_alias=true
  fi
  if [[ "$found_alias" == false ]]; then
    echo -e "  (None configured yet)"
  fi

  echo ""
  if [[ "$all_ok" == true ]]; then
    echo -e "${GREEN}${BOLD}✔ All required offline voice pipeline models are installed and ready.${RESET}"
    return 0
  else
    echo -e "${YELLOW}To download all missing models, run:${RESET}"
    echo -e "  ${BOLD}$0 --all --dir \"${MODELS_DIR}\"${RESET}"
    return 0
  fi
}

# ------------------------------------------------------------------------------
# Download Actions
# ------------------------------------------------------------------------------

download_vad() {
  echo -e "\n${BOLD}=== [1/4] Provisioning Silero VAD v5 ONNX ===${RESET}"
  mkdir -p "$MODELS_DIR"
  local dest="${MODELS_DIR}/${VAD_FILENAME}"
  
  if [[ -f "$dest" ]] && verify_sha256 "$dest" "$VAD_SHA256" 2>/dev/null; then
    echo -e "${GREEN}Silero VAD v5 is already present and valid. Skipping.${RESET}"
  else
    download_file "$VAD_URL" "$dest" "$VAD_BACKUP_URL"
    verify_sha256 "$dest" "$VAD_SHA256"
  fi

  # Create standard directory aliases: silero_vad/ and vad/
  mkdir -p "${MODELS_DIR}/silero_vad"
  ln -sf "../${VAD_FILENAME}" "${MODELS_DIR}/silero_vad/${VAD_FILENAME}"
  mkdir -p "${MODELS_DIR}/vad"
  ln -sf "../${VAD_FILENAME}" "${MODELS_DIR}/vad/${VAD_FILENAME}"
  echo -e "${GREEN}✔ Silero VAD v5 ready at ${dest}${RESET}"
}

download_tts_vi() {
  echo -e "\n${BOLD}=== [2/4] Provisioning Piper Vietnamese TTS ===${RESET}"
  mkdir -p "$MODELS_DIR"
  local archive_path="${MODELS_DIR}/${TTS_VI_ARCHIVE}"
  local target_dir="${MODELS_DIR}/${TTS_VI_DIR_NAME}"
  local onnx_file="${target_dir}/vi_VN-vais1000-medium.onnx"

  if [[ -f "$onnx_file" ]] && [[ -f "${target_dir}/tokens.txt" ]]; then
    echo -e "${GREEN}Piper Vietnamese TTS model is already installed. Skipping.${RESET}"
  else
    download_file "$TTS_VI_URL" "$archive_path"
    verify_sha256 "$archive_path" "$TTS_VI_SHA256"
    extract_archive "$archive_path" "$MODELS_DIR"
    rm -f "$archive_path"
  fi

  # Symlink canonical aliases:
  # - piper -> vits-piper-vi_VN-vais1000-medium
  # - tts-vi -> vits-piper-vi_VN-vais1000-medium
  cd "$MODELS_DIR"
  ln -sfn "$TTS_VI_DIR_NAME" "piper"
  ln -sfn "$TTS_VI_DIR_NAME" "tts-vi"
  cd - >/dev/null

  echo -e "${GREEN}✔ Piper Vietnamese TTS ready at ${target_dir} (aliases: piper, tts-vi)${RESET}"
}

download_tts_en() {
  echo -e "\n${BOLD}=== [3/4] Provisioning Kokoro Multilingual TTS ===${RESET}"
  mkdir -p "$MODELS_DIR"
  local archive_path="${MODELS_DIR}/${TTS_EN_ARCHIVE}"
  local target_dir="${MODELS_DIR}/${TTS_EN_DIR_NAME}"
  local onnx_file="${target_dir}/model.onnx"
  local named_onnx="${target_dir}/kokoro-multi-lang-v1_0.onnx"

  if { [[ -f "$onnx_file" ]] || [[ -f "$named_onnx" ]]; } && [[ -f "${target_dir}/voices.bin" ]]; then
    echo -e "${GREEN}Kokoro Multilingual TTS model is already installed. Skipping.${RESET}"
  else
    download_file "$TTS_EN_URL" "$archive_path"
    verify_sha256 "$archive_path" "$TTS_EN_SHA256"
    extract_archive "$archive_path" "$MODELS_DIR"
    rm -f "$archive_path"
  fi

  # Ensure both model.onnx and kokoro-multi-lang-v1_0.onnx exist via symlink
  if [[ -f "$onnx_file" ]] && [[ ! -e "$named_onnx" ]]; then
    ln -sfn "model.onnx" "$named_onnx"
  elif [[ -f "$named_onnx" ]] && [[ ! -e "$onnx_file" ]]; then
    ln -sfn "kokoro-multi-lang-v1_0.onnx" "$onnx_file"
  fi

  # Symlink canonical aliases:
  # - kokoro -> kokoro-multi-lang-v1_0
  # - tts-en -> kokoro-multi-lang-v1_0
  cd "$MODELS_DIR"
  ln -sfn "$TTS_EN_DIR_NAME" "kokoro"
  ln -sfn "$TTS_EN_DIR_NAME" "tts-en"
  cd - >/dev/null

  echo -e "${GREEN}✔ Kokoro Multilingual TTS ready at ${target_dir} (aliases: kokoro, tts-en)${RESET}"
}

download_stt_vi() {
  echo -e "\n${BOLD}=== [4/4] Provisioning Zipformer Vietnamese STT ===${RESET}"
  mkdir -p "$MODELS_DIR"
  local archive_path="${MODELS_DIR}/${STT_VI_ARCHIVE}"
  local target_dir="${MODELS_DIR}/${STT_VI_DIR_NAME}"
  local enc_file="${target_dir}/encoder.int8.onnx"

  if [[ -f "$enc_file" ]] && [[ -f "${target_dir}/tokens.txt" ]]; then
    echo -e "${GREEN}Zipformer Vietnamese STT model is already installed. Skipping.${RESET}"
  else
    download_file "$STT_VI_URL" "$archive_path"
    verify_sha256 "$archive_path" "$STT_VI_SHA256"
    extract_archive "$archive_path" "$MODELS_DIR"
    rm -f "$archive_path"
  fi

  # Link encoder.onnx and joiner.onnx to their int8 equivalents inside the model directory
  if [[ -f "${target_dir}/encoder.int8.onnx" ]] && [[ ! -e "${target_dir}/encoder.onnx" ]]; then
    ln -sfn "encoder.int8.onnx" "${target_dir}/encoder.onnx"
  fi
  if [[ -f "${target_dir}/joiner.int8.onnx" ]] && [[ ! -e "${target_dir}/joiner.onnx" ]]; then
    ln -sfn "joiner.int8.onnx" "${target_dir}/joiner.onnx"
  fi

  # Symlink canonical aliases:
  # - streaming-zipformer -> sherpa-onnx-zipformer-vi-30M-int8-2026-02-09
  # - stt-vi -> sherpa-onnx-zipformer-vi-30M-int8-2026-02-09
  cd "$MODELS_DIR"
  ln -sfn "$STT_VI_DIR_NAME" "streaming-zipformer"
  ln -sfn "$STT_VI_DIR_NAME" "stt-vi"
  cd - >/dev/null

  echo -e "${GREEN}✔ Zipformer Vietnamese STT ready at ${target_dir} (aliases: streaming-zipformer, stt-vi)${RESET}"
}

download_stt_en() {
  echo -e "\n${BOLD}=== [Optional] Provisioning Zipformer English STT ===${RESET}"
  mkdir -p "$MODELS_DIR"
  local archive_path="${MODELS_DIR}/${STT_EN_ARCHIVE}"
  local target_dir="${MODELS_DIR}/${STT_EN_DIR_NAME}"

  if [[ -d "$target_dir" ]] && [[ -f "${target_dir}/tokens.txt" ]]; then
    echo -e "${GREEN}Zipformer English STT model is already installed. Skipping.${RESET}"
  else
    download_file "$STT_EN_URL" "$archive_path"
    extract_archive "$archive_path" "$MODELS_DIR"
    rm -f "$archive_path"
  fi

  cd "$MODELS_DIR"
  ln -sfn "$STT_EN_DIR_NAME" "stt-en"
  cd - >/dev/null

  echo -e "${GREEN}✔ Zipformer English STT ready at ${target_dir} (alias: ${MODELS_DIR}/stt-en)${RESET}"
}

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
  print_usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      DO_TTS_VI=true
      DO_TTS_EN=true
      DO_STT_VI=true
      DO_VAD=true
      shift
      ;;
    --tts-vi)
      DO_TTS_VI=true
      shift
      ;;
    --tts-en)
      DO_TTS_EN=true
      shift
      ;;
    --stt-vi)
      DO_STT_VI=true
      shift
      ;;
    --stt-en)
      DO_STT_EN=true
      shift
      ;;
    --vad)
      DO_VAD=true
      shift
      ;;
    --check)
      DO_CHECK=true
      shift
      ;;
    --dir|-d)
      if [[ -z "${2:-}" ]]; then
        echo -e "${RED}Error: --dir requires a path argument.${RESET}" >&2
        exit 1
      fi
      MODELS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${RESET}" >&2
      print_usage
      exit 1
      ;;
  esac
done

# Check system dependencies
check_dependencies

# If --check is specified, run check and exit
if [[ "$DO_CHECK" == true ]]; then
  check_models
  exit 0
fi

# If download flags were requested
if [[ "$DO_VAD" == false ]] && [[ "$DO_TTS_VI" == false ]] && [[ "$DO_TTS_EN" == false ]] && [[ "$DO_STT_VI" == false ]] && [[ "$DO_STT_EN" == false ]]; then
  echo -e "${YELLOW}No action requested.${RESET}"
  print_usage
  exit 0
fi

mkdir -p "$MODELS_DIR"
echo -e "${BOLD}Target Directory:${RESET} ${CYAN}${MODELS_DIR}${RESET}"

if [[ "$DO_VAD" == true ]]; then
  download_vad
fi

if [[ "$DO_TTS_VI" == true ]]; then
  download_tts_vi
fi

if [[ "$DO_TTS_EN" == true ]]; then
  download_tts_en
fi

if [[ "$DO_STT_VI" == true ]]; then
  download_stt_vi
fi

if [[ "$DO_STT_EN" == true ]]; then
  download_stt_en
fi

echo ""
echo -e "${GREEN}${BOLD}All requested model downloads completed successfully.${RESET}"
check_models
