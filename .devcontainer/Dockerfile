FROM mcr.microsoft.com/devcontainers/python:3.11

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates jq \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/2026Agents