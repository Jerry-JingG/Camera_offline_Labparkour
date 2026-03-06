# Conda Environment

## Virtual Environment Activation

**CRITICAL**: Before executing ANY Python code or running ANY commands that involve Python in this project, you MUST activate the conda environment.

### Activation Command

```bash
source /home/droplet/anaconda3/etc/profile.d/conda.sh && conda activate parkour
```

### When to Activate

Activate the `parkour` environment before:
- Running Python scripts
- Installing Python packages (pip install)
- Running tests
- Starting development servers
- Executing any Python-related commands

### Command Pattern

Always chain conda activation with your command:

```bash
# CORRECT: Activate first
source /home/droplet/anaconda3/etc/profile.d/conda.sh && conda activate parkour && python script.py

# CORRECT: For multiple commands
source /home/droplet/anaconda3/etc/profile.d/conda.sh && conda activate parkour && pip install package && python script.py

# WRONG: Running without activation
python script.py

# WRONG: Missing conda.sh source
conda activate parkour && python script.py
```

### Verification

After activation, verify the environment:

```bash
source /home/droplet/anaconda3/etc/profile.d/conda.sh && conda activate parkour && which python
```

Should show Python from the `parkour` environment, not system Python.

## Important Notes

- The conda environment persists only for the current command chain
- Each new Bash tool call requires re-activation
- If a command fails with import errors, verify conda environment is activated
- Never assume the environment is already active
