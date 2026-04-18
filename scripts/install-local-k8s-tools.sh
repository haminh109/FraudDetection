#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/.tools/bin"
KIND_VERSION="${KIND_VERSION:-0.29.0}"
HELM_VERSION="${HELM_VERSION:-3.18.4}"

mkdir -p "${BIN_DIR}"

OS="$(uname | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64|amd64) ARCH="amd64" ;;
  arm64|aarch64) ARCH="arm64" ;;
  *)
    echo "Unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac

install_kind() {
  local target="${BIN_DIR}/kind"
  if [[ -x "${target}" ]]; then
    return
  fi

  curl -fsSL -o "${target}" "https://kind.sigs.k8s.io/dl/v${KIND_VERSION}/kind-${OS}-${ARCH}"
  chmod +x "${target}"
}

install_helm() {
  local target="${BIN_DIR}/helm"
  if [[ -x "${target}" ]]; then
    return
  fi

  local tmp_dir=""
  tmp_dir="$(mktemp -d)"
  trap '[[ -n "${tmp_dir}" ]] && rm -rf "${tmp_dir}"' RETURN

  curl -fsSL -o "${tmp_dir}/helm.tgz" "https://get.helm.sh/helm-v${HELM_VERSION}-${OS}-${ARCH}.tar.gz"
  tar -xzf "${tmp_dir}/helm.tgz" -C "${tmp_dir}"
  mv "${tmp_dir}/${OS}-${ARCH}/helm" "${target}"
  chmod +x "${target}"
}

install_kind
install_helm

echo "Installed/available tools in ${BIN_DIR}"
"${BIN_DIR}/kind" version
"${BIN_DIR}/helm" version --short
