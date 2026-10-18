import fcntl

source_txt_path = "temp_task.txt"

f = open(source_txt_path, 'r+')
fcntl.flock(f, fcntl.LOCK_EX)
content = f.readlines()

sorted_content = sorted(content, key=lambda x: int(x.split()[0]))

# Remove the old content
f.seek(0)
f.truncate()

# Write the sorted content
f.writelines(sorted_content)
f.truncate()

# Unlock the file
fcntl.flock(f, fcntl.LOCK_UN)
f.close()