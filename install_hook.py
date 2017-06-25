import sys
import os
from shutil import copyfile
# argv[1] is destination git repo








def main():
    src_path = 'gitgud/commit-msg'
    dest_path = sys.argv[1] + '/.git/hooks/commit-msg'
    dest_samplehook = dest_path + "/commit-msg.sample"
    try:
        os.remove(dest_samplehook)
    except OSError:
        pass
    copyfile(src_path, dest_path)
    return


if __name__ == "__main__":
    main()
