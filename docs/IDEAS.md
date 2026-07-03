# IDEAS & Next-Version Roadmap

> A low-friction place to capture ideas **before** they're committed work. Dump freely in **Inbox**; shape the
> ones worth doing; promote the committed ones to the **vNext roadmap**; when concrete, they graduate into
> `docs/DEVPLAN.md` as milestones/todos.
>
> `DEVPLAN.md` = what we're building **now** (test-backed todos). **This file = what might come next** — no
> commitment, no tests required. Keep the two separate so the plan stays trustworthy and ideas stay cheap.

## How to use (30 seconds)
1. Have an idea? Add a bullet under **📥 Inbox** — one line is fine.
2. Worth pursuing? Move it to **🔨 Shaping** and fill the template (why / scope / risk).
3. Decided to do it next version? Move to **🗺️ vNext roadmap**.
4. Started building it? Create the `Mx`/`Tx` todos in `DEVPLAN.md` and move the idea to **✅ Graduated** (link it).
5. Not doing it? Move to **🧊 Parked** with a one-line reason (so it isn't re-litigated later).

## Idea template
```
### <short title>
- **What:** one line.
- **Why / value:** who benefits, what it unlocks.
- **Scope / effort:** S / M / L.
- **Depends on / risk:** prerequisites, unknowns.
- **Status:** inbox | shaping | vNext | graduated | parked
```

---

## 📥 Inbox
*(raw captures — replace these seed examples with your own)*
- _(seed)_ **MCP server** exposing the KB (candidate queue / trends / parity) to other tools — deferred in `CONTEXT.md`.
- _(seed)_ **RAG beyond human-defined metrics**: let the system pose its own eval queries / self-label once the
  T1.8 golden-set score is stable (ShinkaEvolve's open question).
- _(seed)_ Extend tracking beyond the 5 repos / to other accelerators once the ROCm loop is proven.

## 🔨 Shaping
*(ideas being fleshed out — add the template fields)*

## 🗺️ vNext roadmap (committed for the next version)
*(the shortlist that will become DEVPLAN milestones/todos next)*

## ✅ Graduated (now tracked in DEVPLAN)
*(moved into `docs/DEVPLAN.md` — link the milestone/todo)*

## 🧊 Parked (not now)
- _(seed)_ A2A / heavy multi-agent **product** features — deferred to keep the core loop focused (`CONTEXT.md`).
