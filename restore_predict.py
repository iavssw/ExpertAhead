import re

log_path = "/home/michael/.gemini/antigravity-ide/brain/ccbf1a96-0005-491b-b7af-4bab383c1773/.system_generated/tasks/task-1411.log"
with open(log_path, 'r') as f:
    log_content = f.read()

# Extract the diff block
match = re.search(r'(--- a/src/unified_llm_w4a16_predict/unified_llm_w4a16\.cpp.*?)(?=Terminal ID:|\Z)', log_content, re.DOTALL)
if match:
    diff_content = match.group(1)
    
    # We want to remove the hunk starting with @@ -1986,6 +2073,8 @@ 
    # and also the hunk starting with @@ -2073,11 +2073,6 @@ which was the bad deletion
    
    hunks = re.split(r'(?=@@ -\d+,\d+ \+\d+,\d+ @@)', diff_content)
    
    clean_diff = hunks[0] # header
    for hunk in hunks[1:]:
        if "@@ -1986,6 +2073,8 @@" in hunk or "@@ -2073,11" in hunk:
            continue
        clean_diff += hunk
        
    with open("restore.patch", "w") as f:
        f.write(clean_diff)
        
    print("Patch created.")
else:
    print("Could not find diff in log.")

