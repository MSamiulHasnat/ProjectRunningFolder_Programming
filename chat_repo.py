"""
chat_repo.py — CT-MUSIQ Codebase Chat with Prompt Caching
----------------------------------------------------------
Loads your entire CT-MUSIQ project into Claude's context once,
then caches it so every follow-up question costs almost no tokens.

Usage:
    python chat_repo.py              # chat about the whole project
    python chat_repo.py --file model.py  # focus on one file only

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-...
"""

import anthropic
import os
import argparse

# ── CONFIG ────────────────────────────────────────────────────────────────────

# File types to include in the context
EXTENSIONS = [".py", ".md", ".json", ".txt", ".yaml", ".yml", ".cfg", ".toml"]

# Directories to skip entirely
IGNORE_DIRS = {
    "venv", ".venv", "venv_cuda",           # ← add venv_cuda
    "__pycache__", ".git",
    "node_modules", "dist", "build", ".idea",
    "results", "notebooks",
}

# Files to skip even if they match an extension
IGNORE_FILES = {
    "requirements.txt",  # not interesting for code questions
}

# Model to use — Sonnet 4.6 balances speed and quality well for code chat
MODEL = "claude-sonnet-4-6"

# Max tokens for each reply
MAX_TOKENS = 4096

# ── REPO LOADER ───────────────────────────────────────────────────────────────

def load_repo(root: str = ".", single_file: str = None) -> str:
    """
    Walk the project directory and concatenate all source files into one string.

    If single_file is provided, only that file is loaded (useful for deep dives
    into one module without filling the context with unrelated code).

    Returns a formatted string with file paths as headers, ready to be embedded
    in the system prompt.
    """
    files_content = []

    if single_file:
        # Focus mode: load just one file
        path = os.path.join(root, single_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        files_content.append(f"### FILE: {path}\n{code}")
        return "\n\n".join(files_content)

    # Full repo walk
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place so os.walk skips them entirely
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

        for filename in filenames:
            if filename in IGNORE_FILES:
                continue
            if not any(filename.endswith(ext) for ext in EXTENSIONS):
                continue

            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    code = f.read()
                # Skip empty files
                if not code.strip():
                    continue
                files_content.append(f"### FILE: {filepath}\n{code}")
            except Exception as e:
                print(f"  [skip] {filepath} — {e}")

    return "\n\n".join(files_content)


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

def build_system_prompt(repo_content: str) -> str:
    """
    Build the system prompt that gives Claude full context about the project.

    This is the text that gets cached after the first API call — subsequent
    questions reuse the cache and only charge a tiny fraction of the token cost.
    """
    return f"""You are an expert ML engineer and code assistant with complete knowledge \
of the CT-MUSIQ thesis project described below.

## Project overview
CT-MUSIQ is an architectural adaptation of the MUSIQ Transformer (Ke et al., ICCV 2021) \
for no-reference perceptual image quality assessment of Low-Dose abdominal CT images. \
The dataset is LDCTIQAC 2023 — 1,000 abdominal CT slices (.tiff) with radiologist Likert \
quality scores (0–4), split 700/200/100 train/val/test. Framework: PyTorch. \
GPU: NVIDIA RTX 3060 Laptop (6GB VRAM).

## Key architectural decisions
- 2-scale input pyramid: [224×224, 384×384] → 49 + 144 = 193 patches per image
- 32×32 patch size, grayscale replicated to 3 channels
- Hash-based spatial positional encoding: nn.Embedding for (scale_idx, row_idx, col_idx)
- ViT-B/32 pretrained encoder: d_model=768, 8 heads, 6 layers
- Abdominal CT windowing: width=350, level=40 → HU range [-135, 215] → normalised [0,1]
- Scale-consistency KL divergence loss (lambda=0.1) across per-scale prediction heads
- Mixed precision (fp16) training, batch size=4, two-stage frozen→unfrozen fine-tuning

## Evaluation
PLCC + SROCC + KROCC aggregate score, compared against Lee et al. 2025 leaderboard.

## Your role
- Answer questions about code architecture, logic, bugs, and improvements
- Always reference specific file paths (e.g. model.py, dataset.py) when relevant
- If asked to write or fix code, match the existing style and commenting level
- If asked about a design choice, explain the reasoning in the context of the
  thesis constraints (6GB VRAM, beginner ML level, 1-2 month deadline)
- Never invent values for metrics — only report numbers from actual code

## Full codebase
{repo_content}"""


# ── MAIN CHAT LOOP ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chat with your CT-MUSIQ codebase")
    parser.add_argument(
        "--file", type=str, default=None,
        help="Focus on a single file (e.g. --file model.py)"
    )
    parser.add_argument(
        "--root", type=str, default=".",
        help="Project root directory (default: current directory)"
    )
    args = parser.parse_args()

    # ── Load repo ──────────────────────────────────────────────────────────────
    print("Loading repository...")
    repo_content = load_repo(root=args.root, single_file=args.file)
    word_count = len(repo_content.split())
    char_count = len(repo_content)
    print(f"Loaded ~{word_count:,} words ({char_count:,} chars) from repo")

    if args.file:
        print(f"[Focus mode: {args.file} only]")

    # ── Build system prompt (this gets cached) ─────────────────────────────────
    system_prompt = build_system_prompt(repo_content)

    # ── Anthropic client ───────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n\n"
            "Windows PowerShell:\n"
            '  $env:ANTHROPIC_API_KEY = "sk-ant-..."\n\n'
            "Windows CMD:\n"
            "  set ANTHROPIC_API_KEY=sk-ant-...\n\n"
            "Linux / macOS:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    client = anthropic.Anthropic(api_key=api_key)

    # ── Conversation state ─────────────────────────────────────────────────────
    conversation_history = []
    first_call = True

    print("\n" + "─" * 60)
    print("CT-MUSIQ Codebase Chat")
    print("Repo cached on first question. Type 'exit' to quit.")
    print("─" * 60 + "\n")

    # ── Chat loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        if not user_input:
            continue

        # Append user turn to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # ── API call ───────────────────────────────────────────────────────────
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        # cache_control marks this block for caching.
                        # After the first call, the system prompt is stored
                        # server-side. Every subsequent call reads from cache
                        # at ~10% of the normal input token cost.
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                messages=conversation_history
            )
        except anthropic.APIError as e:
            print(f"\n[API error: {e}]\n")
            # Remove the failed user turn so history stays consistent
            conversation_history.pop()
            continue

        # ── Extract reply ──────────────────────────────────────────────────────
        reply = response.content[0].text

        # Append assistant turn to history
        conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        # ── Token usage display ────────────────────────────────────────────────
        usage = response.usage
        cache_write  = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read   = getattr(usage, "cache_read_input_tokens", 0) or 0
        normal_input = usage.input_tokens or 0
        output_toks  = usage.output_tokens or 0

        if first_call and cache_write > 0:
            cache_status = f"written ({cache_write:,} tokens cached)"
            first_call = False
        elif cache_read > 0:
            cache_status = f"hit ({cache_read:,} tokens from cache — cheap!)"
        else:
            cache_status = "miss"

        print(f"\nClaude: {reply}")
        print(
            f"\n[Cache: {cache_status} | "
            f"Input: {normal_input:,} | "
            f"Output: {output_toks:,}]\n"
        )


if __name__ == "__main__":
    main()