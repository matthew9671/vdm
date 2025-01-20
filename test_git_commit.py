import subprocess

def get_git_commit_message():
    # Run the Git command to get the latest commit message
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        cwd="/home/yixiuz/vdm/",
    )
    return result.stdout.strip()  # Return the commit message

if __name__ == "__main__":
    print(get_git_commit_message())