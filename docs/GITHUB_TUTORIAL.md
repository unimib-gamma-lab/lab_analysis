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

Pick **one** of the three options.

**Option A – GitHub Desktop (graphical, Windows and macOS, easiest for beginners)**

1. Install it from <https://desktop.github.com>.
2. *File → Options → Accounts → Sign in* to GitHub.com (it opens your browser).

**Option B – GitHub CLI (terminal, any system)**

1. Install it: `winget install --id GitHub.cli` (Windows) · `brew install gh` (macOS) ·
   `sudo dnf install gh` (Fedora) · other systems: <https://cli.github.com>.
2. Run `gh auth login` and answer: *GitHub.com* → *HTTPS* → *Yes* (authenticate Git with your
   GitHub credentials) → *Login with a web browser*.
3. Check with `gh auth status`: it should say you are logged in to github.com.

**Option C – SSH key (only if you already have an SSH key added to your GitHub account)**

1. Test the connection: `ssh -T git@github.com`. The expected answer is
   `Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.`
2. Clone with the SSH address (see section 3) instead of the `https://` one.

> The address you clone with decides how you log in: `https://...` uses Options A/B,
> `git@github.com:...` uses Option C. A repository already cloned with one of them can be switched
> with `git remote set-url origin <new-address>`.

> GitHub **does not accept your account password** from Git any more. If Git asks for a password
> and rejects it, you skipped this step (see section 9).

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

(With Option B you can also use `gh repo clone unimib-gamma-lab/lab_analysis`. With Option C clone
`git@github.com:unimib-gamma-lab/lab_analysis.git` instead.)

Check that it worked:

```bash
git status      # should start with "On branch main"
ls -a           # should list .gitignore and .python-version among the files
```

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
- Don't copy files into the repository with a file manager without checking `ls -a` afterwards:
  hidden files such as `.gitignore` are easily left behind (see section 9).

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

---

## 9. Common error messages

Git and uv translate part of their messages into your system language. If yours are, for example,
in Italian, look for the equivalent sentence. On Linux and macOS you can put `LANG=C` before a
command (e.g. `LANG=C git status`) to get English messages, which are easier to search online.

### Environment (uv, Python)

| Message / symptom | Cause | What to do |
|---|---|---|
| `No interpreter found for Python 3.12 ... Python downloads are set to 'manual'` | uv is configured not to download Python on its own. | Run `uv python install 3.12` (use the version written in `.python-version`), then `uv sync` again. |
| `uv: command not found` or `git: command not found` | The program is not installed, or the terminal was opened before the installation. | Install it (see the README) and **reopen the terminal**. |
| `ModuleNotFoundError` after a `git pull` | A teammate added a new library. | Run `uv sync`. |
| Strange import errors even after `uv sync` | The `.venv/` folder is damaged. | Delete it and recreate it: `rm -rf .venv && uv sync` (Windows PowerShell: `Remove-Item -Recurse -Force .venv`, then `uv sync`). Nothing is lost: `.venv/` is rebuilt from `uv.lock`. |
| `.venv/` shows up in `git status`, or was committed | `.gitignore` is missing (hidden files not copied). | Check `ls -a`. If `.gitignore` is missing, ask. If `.venv/` was already committed: `git rm -r --cached .venv`, then commit. |

### Login and push

| Message / symptom | Cause | What to do |
|---|---|---|
| `Permission denied (publickey)` | SSH key not added to your GitHub account, or not loaded. | Add the key under GitHub → *Settings → SSH and GPG keys*, test with `ssh -T git@github.com`; or use Option A/B with the `https://` address. |
| `Authentication failed` / `Support for password authentication was removed` | GitHub does not accept the account password from Git. | Log in as described in section 2.4 (Option A or B). |
| `Author identity unknown` / `Please tell me who you are` | Name and email not configured. | Run the commands in section 2.3, then repeat the commit. |
| `rejected ... (fetch first)` or `non-fast-forward` | Someone pushed before you. | `git pull`, solve conflicts if any (section 5), then `git push`. **Never** `--force`. |
| `You are not currently on a branch` or `HEAD detached` | Your repository is not on a branch. | Run `git status`. If it doesn't start with "On branch ...", **stop and ask the group**: don't try to fix it by guessing. |
| `File ... exceeds GitHub's file size limit of 100 MB` | A file is too large for GitHub. | If you haven't pushed: `git reset --soft HEAD~1`, then `git restore --staged <file>`, commit again without it and talk to the group about where to store it. |

### Commit

| Message / symptom | Cause | What to do |
|---|---|---|
| `nothing to commit` | Git sees no changes to save. | Run `git status`. If files appear under *Untracked files* or *Changes not staged*, use `git add <file>` first. If nothing appears, you are probably in the wrong folder (`pwd`) or the files were never copied there. |
| Hidden files (`.gitignore`, `.python-version`) missing after copying files by hand | File managers hide files starting with a dot. | Copy from the terminal with `cp -a source/. destination/`, or show hidden files (Linux: `Ctrl+H`; macOS: `Cmd+Shift+.`; Windows: *View → Show → Hidden items*). |
