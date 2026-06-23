# Gitee Core Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clean minimal FunASR-GGML source snapshot to `https://gitee.com/hua1848/asr`.

**Architecture:** Build a separate release workspace under `/tmp/asr-release` instead of relying on the current broken `.git` directory. Copy only the C++ engine, CLI/SDK, Web app tools, browser extension, source tests, essential docs, and vendored ggml source; exclude outputs, model binaries, local media, credentials, caches, and build artifacts.

**Tech Stack:** C++17, CMake, GGML, Python FastAPI tools, Chrome extension files, Git.

---

### Task 1: Prepare Release Metadata

**Files:**
- Modify: `.gitignore`
- Create: `docs/release_scope.md`

- [ ] **Step 1: Expand `.gitignore` for public release safety**

Add ignore entries for build outputs, run outputs, local media, cookies, caches, CodeGraph state, and Windows zone identifier files.

- [ ] **Step 2: Add release scope document**

Create `docs/release_scope.md` explaining what is included and excluded from the Gitee release.

### Task 2: Assemble Clean Release Tree

**Files:**
- Create directory: `/tmp/asr-release`

- [ ] **Step 1: Copy selected source directories**

Copy `core`, `model`, `compute`, `pipeline`, `cli`, `sdk`, `include`, `examples`, `tools`, `browser-extension`, `test`, `third_party`, and selected docs into `/tmp/asr-release`.

- [ ] **Step 2: Exclude noisy files**

Ensure `/tmp/asr-release` does not contain `outputs`, `build-cuda`, model binaries, cookies, local audio/video/subtitle samples, caches, generated test executables, or CodeGraph files.

### Task 3: Verify Release Snapshot

**Files:**
- Inspect: `/tmp/asr-release`

- [ ] **Step 1: Search for oversized files**

Run `find /tmp/asr-release -type f -size +10M -print`.

- [ ] **Step 2: Search for sensitive files**

Run checks for `cookies`, `DEEPSEEK_API_KEY`, `.codegraph`, `outputs`, `build-cuda`, and media sample extensions.

- [ ] **Step 3: Confirm core files are present**

Confirm `CMakeLists.txt`, `README.md`, `cli/funasr_cli.cpp`, `tools/funasr_web_app.py`, and `browser-extension/bilibili-funasr-sidebar/manifest.json` exist.

### Task 4: Push to Gitee

**Files:**
- Git repo: `/tmp/asr-release`

- [ ] **Step 1: Initialize git and commit**

Run `git init`, set user identity if needed, add all files, and commit with message `Initial core FunASR-GGML release`.

- [ ] **Step 2: Add Gitee remote**

Set `origin` to `https://gitee.com/hua1848/asr`.

- [ ] **Step 3: Push**

Push the release branch to Gitee. If authentication or network access blocks the push, report the exact blocker and leave the prepared repo ready at `/tmp/asr-release`.

### Task 5: Final Verification

**Files:**
- Git repo: `/tmp/asr-release`

- [ ] **Step 1: Verify local git status**

Run `git status --short` in `/tmp/asr-release`.

- [ ] **Step 2: Verify remote state if push succeeds**

Run `git ls-remote origin HEAD` or report why it cannot be checked.
