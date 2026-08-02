---
name: smithers-remove
description: Remove an installed workflow pack. Run `smithers remove --help` for usage details.
requires_bin: smithers
command: smithers remove
---

# smithers remove

Remove an installed workflow pack.

## Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | `string` | yes | Installed pack name |

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--global` | `boolean` | `false` | Remove from ~/.smithers/packs |
