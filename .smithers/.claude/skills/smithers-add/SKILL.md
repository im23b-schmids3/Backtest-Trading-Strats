---
name: smithers-add
description: Install a workflow pack from GitHub, npm, or a local file. Run `smithers add --help` for usage details.
requires_bin: smithers
command: smithers add
---

# smithers add

Install a workflow pack from GitHub, npm, or a local file.

## Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `spec` | `string` | yes | GitHub, npm, or file pack spec |

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--global` | `boolean` | `false` | Install in ~/.smithers/packs instead of the local project |
| `--yes` | `boolean` | `false` | Skip trust confirmation |
