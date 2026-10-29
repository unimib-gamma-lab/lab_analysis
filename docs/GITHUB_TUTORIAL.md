# GitHub tutorial for the group

Everything you need to work on this repository without ever having used Git or GitHub.
Read sections 1–4 once, then keep sections 7–8 at hand as a cheat sheet.

Placeholders in `<angle brackets>` must be replaced with your own values.

---

## 1. The big picture

| Term | Meaning |
|---|---|
| **Git** | A program on your computer that records the history of every change to the files of a project. |
| **GitHub** | A website that hosts the shared copy of the project, so that we can exchange our work. |
| **Repository (repo)** | The project folder together with its full history. |
| **Clone** | Download the repository from GitHub to your computer (done once). |
| **Commit** | A saved snapshot of your changes, with a short message describing them. |
| **Push** | Upload your commits to GitHub. |
| **Pull** | Download the commits made by others from GitHub. |
| **Branch** | A parallel line of work: your unfinished changes don't disturb anyone else. |
| **Pull Request (PR)** | A proposal to merge your branch into `main`; a teammate reviews it first. |

Mental model:

```
files you edit  --add + commit-->  your local history  --push-->  GitHub  --pull-->  teammates
```

Nothing you do on your computer reaches the others until you **push**, and nothing they do reaches
you until you **pull**.

---

## 2. One-time setup

### 2.1 GitHub account and invitation

1. Create an account on <https://github.com> (if you don't have one).
2. Send your username to `<owner>`, who will invite you to the organization `unimib-gamma-lab`.
3. Accept the invitation (you receive an email, or look at the notifications on github.com).

### 2.2 Install Git

See the table in the [README](../README.md#1-install-git). Check with:

```bash
git --version
```

### 2.3 Tell Git who you are

Use the same email as your GitHub account:

```bash
git config --global user.name "Name Surname"
git config --global user.email "your-github-email@example.com"
git config --global pull.rebase false
```

### 2.4 Log in to GitHub from your computer

Pick **one** of the two options.

**Option A – GitHub Desktop (graphical, Windows and macOS, easiest for beginners)**

1. Install it from <https://desktop.github.com>.
2. *File → Options → Accounts → Sign in* to GitHub.com (it opens your browser).

**Option B – GitHub CLI (terminal, any system)**

1. Install it: `winget install --id GitHub.cli` (Windows) · `brew install gh` (macOS) ·
   `sudo dnf install gh` (Fedora) · other systems: <https://cli.github.com>.
2. Run `gh auth login` and answer: *GitHub.com* → *HTTPS* → *Yes* (authenticate Git with your
   GitHub credentials) → *Login with a web browser*.

### 2.5 Install uv

See the [README](../README.md#2-install-uv).

> Tip: do **not** clone the repository inside a folder synchronised by OneDrive, Dropbox, iCloud or
> Google Drive. Those tools can corrupt the hidden `.git` folder. Use a plain folder such as
> `~/lab/` or `C:\Users\<you>\lab\`.

---

## 3. Get the repository (once)

**Terminal**

```bash
git clone https://github.com/unimib-gamma-lab/lab_analysis.git
cd lab_analysis
uv sync
```

(With Option B you can also use `gh repo clone unimib-gamma-lab/lab_analysis`.)

**GitHub Desktop**

*File → Clone repository → choose `unimib-gamma-lab/lab_analysis` → Clone*. Then open a terminal in that folder
(*Repository → Open in Terminal*) and run `uv sync`.

---

## 4. Daily workflow

**Golden rule: pull before you start, push when you finish.**

### 4.1 Start a task

```bash
git switch main
git pull                                  # get the latest work of the others
git switch -c <yourname>/<short-topic>    # create your own branch, e.g. ale/compton-calibration
```

You create the branch **once per task**. Then work normally: edit notebooks, write scripts, etc.

### 4.2 Save your work (commit)

```bash
git status                                # what changed? (use it all the time)
git add analysis/compton/calibration.ipynb    # choose the files to include
git commit -m "Add energy calibration of the detector"
```

- Commit small and often: one logical change per commit.
- Write messages in the imperative and say *what* changed: "Fix angle offset in Compton fit",
  not "changes" or "asdf".
- `git add .` adds everything that changed: handy, but check `git status` first so you don't
  commit files you didn't mean to.

### 4.3 Upload it (push)

The first time on a new branch:

```bash
git push -u origin <yourname>/<short-topic>
```

Afterwards, just `git push`.

### 4.4 Open a Pull Request

1. After the push, the terminal prints a link; or go to the repository on github.com and click the
   yellow banner **Compare & pull request**.
2. Write a title and a short description (what you did, what to check).
3. On the right, set a teammate as **Reviewer**, then click **Create pull request**.
4. The reviewer reads the changes (tab *Files changed*), comments or approves, and clicks
   **Merge pull request**.
5. Everyone then updates their computer:

```bash
git switch main
git pull
```

If you want to keep working on the same topic, create a new branch from the updated `main`.

### 4.5 Same thing in GitHub Desktop

| Step | In GitHub Desktop |
|---|---|
| Pull | *Fetch origin* → *Pull origin* (top bar) |
| New branch | *Current Branch → New Branch* |
| Commit | *Changes* tab: tick the files, write a summary, click *Commit to `<branch>`* |
| Push | *Push origin* (or *Publish branch* the first time) |
| Pull Request | *Create Pull Request* (opens github.com) |

---

## 5. Merge conflicts

A conflict happens when two people changed the same lines of the same file. Git doesn't guess: it
stops and marks the file like this:

```
<<<<<<< HEAD
your version of the lines
=======
the other person's version
>>>>>>> main
```

What to do:

1. Open the file, decide what the final text should be (yours, theirs, or a mix) and **delete the
   three marker lines** (`<<<<<<<`, `=======`, `>>>>>>>`).
2. Save, then `git add <file>` and `git commit`.
3. If you are not sure, stop and ask the group. A conflict is never an emergency.

How to avoid most of them:

- Pull often and keep branches short-lived.
- Only one person at a time works on a given notebook.
- Don't reformat or rename files "just to tidy up" while others are working on them.

---

## 6. Rules of the group

**Do**

- Create one branch per task and merge it through a Pull Request.
- Run `git status` before every commit.
- Run `uv sync` after a pull if `pyproject.toml` or `uv.lock` changed.
- Keep raw data in `data/raw/` untouched.

**Don't**

- Don't commit `.venv/`, passwords or tokens (`.gitignore` already excludes `.venv/`).
- Don't commit files larger than ~50 MB (GitHub blocks anything above 100 MB).
- Don't use `git push --force`, ever, unless the whole group agrees.
- Don't work directly on `main`.

---

## 7. Cheat sheet

| I want to... | Command |
|---|---|
| See what changed | `git status` |
| See the exact changes | `git diff` |
| Get the others' latest work | `git pull` |
| Create a branch and switch to it | `git switch -c <name>` |
| Switch to an existing branch | `git switch <name>` |
| Stage a file | `git add <file>` |
| Commit | `git commit -m "message"` |
| Upload my commits | `git push` |
| See the history | `git log --oneline` |
| Download a repo | `git clone <url>` |

---

## 8. Something went wrong?

| Situation | What to do |
|---|---|
| I committed, but haven't pushed, and want to undo the commit **keeping my changes** | `git reset --soft HEAD~1` |
| I want to throw away my uncommitted changes to a file | `git restore <file>` (**irreversible**: the changes are lost) |
| `git pull` says my local changes would be overwritten | Commit your changes first (on your branch), then pull again. |
| I'm on the wrong branch | `git status` shows it; commit or ask before switching. |
| I have no idea what's happening | Run `git status`, read it, and send the output to the group. |

When in doubt, **ask before running a command you don't understand**: almost everything in Git
is recoverable, as long as the work was committed.
