import os


def remove_old(files_to_remove):
    open("train.log", "w").close()
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑 Deleted: {f}")
