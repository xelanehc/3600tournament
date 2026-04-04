import os
from multiprocessing import Process, Queue

from game.board import Board


def get_file_permissions(file_path):
    import stat

    """
    Get file permissions in both symbolic and octal formats.
    """

    # Ensure file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file status
    file_stat = os.stat(file_path)

    # Get octal permission mask
    octal_perm = oct(file_stat.st_mode & 0o777)

    # Get symbolic permission string (e.g., -rw-r--r--)
    symbolic_perm = stat.filemode(file_stat.st_mode)

    return symbolic_perm, octal_perm


def drop_priveliges(user_name=None, group_name=None):
    import grp
    import os
    import pwd

    if not user_name is None and not group_name is None:
        uid = pwd.getpwnam(user_name).pw_uid
        gid = grp.getgrnam(group_name).gr_gid

        # print(uid, gid)
        os.setgid(gid)
        os.setuid(uid)


def apply_seccomp():
    try:
        import seccomp
    except ImportError:
        import pyseccomp as seccomp
    import os
    import signal

    import prctl

    prctl.set_ptracer(None)
    prctl.set_no_new_privs(True)
    ctx = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
    # filesystem
    ctx.add_rule(seccomp.KILL, "chdir")
    ctx.add_rule(seccomp.KILL, "chmod")
    ctx.add_rule(seccomp.KILL, "fchmod")
    ctx.add_rule(seccomp.KILL, "fchmodat")
    ctx.add_rule(seccomp.KILL, "chown")
    ctx.add_rule(seccomp.KILL, "fchown")
    ctx.add_rule(seccomp.KILL, "lchown")
    ctx.add_rule(seccomp.KILL, "chroot")
    # ctx.add_rule(seccomp.KILL, 'unlink')
    # ctx.add_rule(seccomp.KILL, 'unlinkat')
    # ctx.add_rule(seccomp.KILL, 'rename')
    # ctx.add_rule(seccomp.KILL, 'renameat')
    # ctx.add_rule(seccomp.KILL, 'rmdir')
    # ctx.add_rule(seccomp.KILL, 'mkdir')
    ctx.add_rule(seccomp.KILL, "mount")
    ctx.add_rule(seccomp.KILL, "umount2")
    ctx.add_rule(seccomp.KILL, "symlink")
    # ctx.add_rule(seccomp.KILL, 'link')
    # ctx.add_rule(seccomp.KILL, 'creat')
    ctx.add_rule(seccomp.KILL, "truncate")
    ctx.add_rule(seccomp.KILL, "ftruncate")
    # ctx.add_rule(seccomp.KILL, 'pwrite64')

    # #time
    ctx.add_rule(seccomp.KILL, "adjtimex")
    ctx.add_rule(seccomp.KILL, "clock_settime")
    ctx.add_rule(seccomp.KILL, "clock_adjtime")
    ctx.add_rule(seccomp.KILL, "settimeofday")

    # #network
    ctx.add_rule(seccomp.KILL, "socket")
    ctx.add_rule(seccomp.KILL, "bind")
    ctx.add_rule(seccomp.KILL, "accept")
    ctx.add_rule(seccomp.KILL, "connect")
    ctx.add_rule(seccomp.KILL, "listen")
    ctx.add_rule(seccomp.KILL, "setsockopt")
    ctx.add_rule(seccomp.KILL, "getsockopt")
    ctx.add_rule(seccomp.KILL, "sendto")
    ctx.add_rule(seccomp.KILL, "recvfrom")
    ctx.add_rule(seccomp.KILL, "sendmsg")
    ctx.add_rule(seccomp.KILL, "recvmsg")
    ctx.add_rule(seccomp.KILL, "unshare")

    # kernel
    ctx.add_rule(seccomp.KILL, "reboot")
    ctx.add_rule(seccomp.KILL, "shutdown")
    ctx.add_rule(seccomp.KILL, "sysfs")
    ctx.add_rule(seccomp.KILL, "sysinfo")
    ctx.add_rule(seccomp.KILL, "delete_module")
    ctx.add_rule(seccomp.KILL, "prctl")
    ctx.add_rule(seccomp.KILL, "execve")
    ctx.add_rule(seccomp.KILL, "execveat")
    ctx.add_rule(seccomp.KILL, "seccomp")

    # #i/o
    # ctx.add_rule(seccomp.KILL, 'ioctl')
    # ctx.add_rule(seccomp.KILL, 'keyctl')
    # ctx.add_rule(seccomp.KILL, 'perf_event_open')
    ctx.add_rule(seccomp.KILL, "kexec_load")
    # ctx.add_rule(seccomp.KILL, 'iopl')
    # ctx.add_rule(seccomp.KILL, 'ioperm')

    # process limiting + scheduling
    ctx.add_rule(seccomp.KILL, "exit")
    ctx.add_rule(seccomp.KILL, "setuid")
    ctx.add_rule(seccomp.KILL, "setgid")
    ctx.add_rule(seccomp.KILL, "capset")
    ctx.add_rule(seccomp.KILL, "capget")
    ctx.add_rule(seccomp.KILL, "kill")
    ctx.add_rule(seccomp.KILL, "tkill")
    ctx.add_rule(seccomp.KILL, "tgkill")
    ctx.add_rule(seccomp.KILL, "setrlimit")
    ctx.add_rule(seccomp.KILL, "setpriority")
    ctx.add_rule(seccomp.KILL, "sched_setparam")
    ctx.add_rule(seccomp.KILL, "sched_setscheduler")

    ctx.load()


# starts up a player process ready to recieve instructions
def run_player_process(
    player_name,
    submission_dir,
    player_queue,
    return_queue,
    limit_resources,
    use_gpu,
    out_queue,
    user_name=None,
    group_name=None,
):
    # try:
    import importlib
    import os
    import sys
    import time
    import traceback

    import psutil

    sys.path.append(submission_dir)

    if use_gpu:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

    limit_mb = 1536
    limit_bytes = limit_mb * 1024 * 1024  # set limit to 1 gb

    def checkMemory():
        pid = os.getpid()
        process = psutil.Process(pid)

        total_memory = process.memory_info().rss

        for child in process.children(recursive=True):
            total_memory += child.memory_info().rss

        if limit_resources and total_memory > limit_bytes:
            raise MemoryError("Allocated too much memory on physical RAM")

        return total_memory

    # Set your VRAM limit in bytes
    vram_limit_bytes = 4 * 1024**3  # 4 GB

    def checkVRAM():
        if use_gpu:
            pid = os.getpid()

            # Get current process + all child PIDs
            process = psutil.Process(pid)
            pids = [process.pid] + [
                child.pid for child in process.children(recursive=True)
            ]

            total_vram = 0
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid in pids:
                    total_vram += proc.usedGpuMemory  # in bytes

            if limit_resources and total_vram > vram_limit_bytes:
                raise MemoryError("Allocated too much VRAM on GPU")

            return total_vram
        return 0

    def get_cur_time():
        return time.perf_counter()

    if limit_resources:
        import resource

        resource.setrlimit(
            resource.RLIMIT_RSS, (limit_bytes, limit_bytes)
        )  # only allow current process to run

        drop_priveliges(user_name, group_name)
        apply_seccomp()
    else:

        class QueueWriter:
            def __init__(self, queue):
                self.queue = queue
                self.turn = ""

            def set_turn(self, t):
                self.turn = t

            def write(self, message):
                # This method is called by print, we send message to out_queue
                if message != "\n":  # Ignore empty newlines that can be printed
                    self.queue.put(
                        "".join(["[", player_name, " | ", self.turn, "]: ", message])
                    )

            def flush(self):
                pass

        printer = QueueWriter(out_queue)
        sys.stdout = printer

    try:
        importlib.import_module(player_name)
        module = importlib.import_module(player_name + ".agent")
    except ModuleNotFoundError:
        print(f"Error: The module {player_name} was not found.")
        print(traceback.format_exc())
        return
    except ImportError as e:
        print(f"Error during import of {player_name}: {e}")
        print(traceback.format_exc())
        return
    except Exception as e:
        print(f"An unexpected error occurred: {player_name}, {e}")
        print(traceback.format_exc())
        return

    player = None
    start = 0
    stop = 0
    return_queue.put(True)

    while True:
        func = player_queue.get()

        # called to play a turn
        if func == "play":
            try:
                temp_board, rat_samples, time_left = player_queue.get()
                if not limit_resources:
                    printer.set_turn(f"turn #{temp_board.turn_count}")

                try:
                    start = get_cur_time()

                    def time_left_func():
                        return time_left - (get_cur_time() - start)

                    # print("playing")
                    player_move = player.play(
                        temp_board, rat_samples, time_left_func
                    )
                    # print("return from play")
                    stop = get_cur_time()
                except:
                    print(traceback.format_exc())
                    return_queue.put((None, -1, traceback.format_exc()))
                    continue

                try:
                    checkMemory()
                except MemoryError:
                    print(traceback.format_exc())
                    return_queue.put(("Memory", -1, traceback.format_exc()))
                    continue

                try:
                    checkVRAM()
                except MemoryError:
                    print(traceback.format_exc())
                    return_queue.put(("GPU VRAM", -1, traceback.format_exc()))
                    continue

                return_queue.put((player_move, stop - start, ""))

                # print(return_queue.qsize())
            except:
                return_queue.put(("Fail", -1, traceback.format_exc()))

        # called to construct the player class
        elif func == "construct":
            try:
                temp_board, transition_matrix, time_left = player_queue.get()

                if not limit_resources:
                    printer.set_turn("construct")

                try:
                    start = get_cur_time()

                    def time_left_func():
                        return time_left - (get_cur_time() - start)

                    player = module.PlayerAgent(temp_board, transition_matrix, time_left_func)

                    stop = get_cur_time()
                except:
                    return_queue.put((False, -1, traceback.format_exc()))
                    continue

                try:
                    checkMemory()
                except MemoryError:
                    print(traceback.format_exc())
                    return_queue.put(("Memory", -1, traceback.format_exc()))
                    continue

                return_queue.put((True, stop - start, ""))
            except:
                print(traceback.format_exc())
                return_queue.put(("Fail", -1, traceback.format_exc()))

        elif func == "commentary":
            try:
                if not limit_resources:
                    printer.set_turn("commentate")

                try:
                    message = player.commentate()
                except:
                    print(traceback.format_exc())
                    return_queue.put("commentary failed")
                    continue

                return_queue.put(message)
            except:
                print(traceback.format_exc())
                return_queue.put("commentary failed")

class PlayerProcess:
    def __init__(
        self,
        is_player_a,
        player_name,
        submission_dir,
        player_queue,
        return_queue,
        limit_resources,
        use_gpu,
        out_queue,
        user_name=None,
        group_name=None,
    ):
        self.process = Process(
            target=run_player_process,
            args=(
                player_name,
                submission_dir,
                player_queue,
                return_queue,
                limit_resources,
                use_gpu,
                out_queue,
                user_name,
                group_name,
            ),
        )
        self.player_queue = player_queue
        self.return_queue = return_queue
        self.is_player_a = is_player_a
        self.player_name = player_name
        self.limit_resources = limit_resources

    def start(self):
        self.process.start()

    # runs player construct command
    def run_timed_constructor(self, game_board, timeout, extra_ret_time, transition_matrix=None):
        # import threading

        # finished = threading.Event()
        # checker_thread = threading.Thread(target=check_process, args=(process, finished, return_queue, False))
        # checker_thread.start()

        self.player_queue.put("construct")
        temp_board = game_board.get_copy(False)
        self.player_queue.put((temp_board, transition_matrix, timeout))

        try:
            ok, timer, message = self.return_queue.get(
                block=True, timeout=timeout + extra_ret_time
            )
            # finished.set()

            if ok == False:
                print(f"{self.player_name}: Constructor failed.\n {message}")
                return False, message
            if ok == "Memory" and timer == -1:
                print(f"{self.player_name}: Memory error.\n {message}")
                return False, message
            if ok == "Fail" and timer == -1:
                raise RuntimeError(
                    f"{self.player_name}: Something went wrong while running player constructor.\n {message}"
                )

            return timer < timeout, message
        except:
            # finished.set()
            return False, "Timeout"

    # runs player play command
    def run_timed_play(self, game_board, rat_samples, timeout, extra_ret_time):
        # print("running timed play")
        temp_board = game_board.get_copy(False)

        self.player_queue.put("play")
        self.player_queue.put((temp_board, rat_samples, timeout))

        try:
            # print("waiting for move")
            move, timer, message = self.return_queue.get(
                block=True, timeout=timeout + extra_ret_time
            )

            # print("return")
            # print(moves, timer, message)

            if move == None:
                print("Player code caused exception")
                return None, -1, message
            if move == "Memory" and timer == -1:
                print("Memory error")
                return None, -2, message
            if move == "Fail" and timer == -1:
                raise RuntimeError(
                    f"Something went wrong while running player move. \n{message}"
                )

            if timer < timeout:
                return move, timer, message
            return None, timeout, "Timeout"
        except:
            return None, -1, "Timeout"
        
    def run_timed_commentary(self, timeout, extra_ret_time=0):

        self.player_queue.put("commentary")
        try:
            message = self.return_queue.get(
                block=True, timeout=timeout + extra_ret_time
            )
            if(not isinstance(message, str)):
                return "commentary failed"
            
            return message
        except:
            return "commentary failed"

    def terminate_process_and_children(self):
        import psutil

        # Find the process by PID
        pid = self.process.pid
        parent_process = None
        children = None
        try:
            parent_process = psutil.Process(pid)
        except psutil.NoSuchProcess as e:
            print(f"Process has already been closed.")

        if not parent_process is None:
            children = parent_process.children(recursive=True)

        # Kill the parent process
        if not parent_process is None and parent_process.is_running():
            try:
                parent_process.terminate()
            except psutil.NoSuchProcess as e:
                print(f"Process has already been closed.")
            except Exception as e:
                print(f"Error while killing process: {e}")

        if not children is None:
            for child in children:
                if child.is_running():
                    try:
                        child.terminate()

                    except psutil.NoSuchProcess as e:
                        print(f"Process  does not exist.")
                    except Exception as e:
                        print(f"Error while killing process: {e}")

        if not parent_process is None and parent_process.is_running():
            try:
                parent_process.kill()
            except psutil.NoSuchProcess:
                print(f"Process  does not exist.")
            except Exception as e:
                print(f"Error while killing process: {e}")

        if not children is None:
            for child in children:
                if child.is_running():
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        print(f"Process  does not exist.")
                    except Exception as e:
                        print(f"Error while killing process: {e}")

    def pause_process_and_children(self):
        # Find the process by PID
        if self.limit_resources:
            import os
            import signal
            import time

            import psutil

            try:
                pid = self.process.pid
                parent_process = psutil.Process(pid)

                children = parent_process.children(recursive=True)

                # send sigstop to parent process
                if parent_process.is_running():
                    try:
                        os.kill(pid, signal.SIGSTOP)
                    except psutil.NoSuchProcess:
                        print(f"Process  does not exist.")
                    except Exception as e:
                        print(f"Error while killing process: {e}")

                i = 0
                while parent_process.status() == psutil.STATUS_RUNNING and i < 50:
                    time.sleep(0.001)
                    i += 1
                if parent_process.status() == psutil.STATUS_RUNNING:
                    os.kill(pid, signal.SIGKILL)

                for child in children:
                    if child.is_running():
                        try:
                            os.kill(child.pid, signal.SIGSTOP)
                        except psutil.NoSuchProcess:
                            print(f"Process  does not exist.")
                        except Exception as e:
                            print(f"Error while killing process: {e}")

                for child in children:
                    i = 0
                    while child.status() == psutil.STATUS_RUNNING and i < 50:
                        time.sleep(0.001)
                        i += 1
                    if child.status() == psutil.STATUS_RUNNING:
                        os.kill(child.pid, signal.SIGKILL)

            except:
                print("error pausing processes")

    def restart_process_and_children(self):
        if self.limit_resources:
            import os
            import signal
            import time

            import psutil

            pid = self.process.pid
            parent_process = psutil.Process(pid)

            children = parent_process.children(recursive=True)

            try:
                for child in children:
                    if child.is_running():
                        try:
                            os.kill(child.pid, signal.SIGCONT)
                        except psutil.NoSuchProcess:
                            print(f"Process does not exist.")
                        except Exception as e:
                            print(f"Error while killing process: {e}")

                for child in children:
                    i = 0
                    while child.status() != psutil.STATUS_STOPPED and i < 50:
                        time.sleep(0.001)
                        i += 1

                # send sigstop to parent process
                if parent_process.is_running():
                    try:
                        os.kill(pid, signal.SIGCONT)
                    except psutil.NoSuchProcess:
                        print(f"Process does not exist.")
                    except Exception as e:
                        print(f"Error while killing process: {e}")

                i = 0
                while parent_process.status() == psutil.STATUS_STOPPED and i < 50:
                    time.sleep(0.001)
                    i += 1

            except:
                print("error restarting processes")
