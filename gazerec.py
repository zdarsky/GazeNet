from tkinter import Tk, Canvas
import time
import csv
import cv2
import threading
import numpy as np
import logging
import os


logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    level=logging.DEBUG)


#  len(chars) = 36 for representing 2^31 states in 6 chars
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
def get_enc_time() -> str:
    n = int(time.time())
    s = ""
    while n > 0:
        s += CHARS[n % len(CHARS)]
        n //= len(CHARS)
    return s[::-1]  # reverse

cam_config = {
    "cam_res": (1920, 1080),
    "cam_fps": 30.0,
    "crop": (750, 1250, 400, 1000),  # x1, x2, y1, y2
    "max_dur": 3000
}

line_config = {
    "speed_start": 5,
    "padding": 50,
    # "step" : 90
    "step": 170
}

dot_config = {
    "number_dots": 2000,
    "dot_big_dur": 0.3,
    "dot_small_dur": 0.5
}

#frame = frame[200:800, 750:1250]


dot_x = None
dot_y = None
cap = None

logging.info("test start")

class Gui:
    def __init__(self, master):
        self.master = master
        master.title("Gazerec")


root = Tk()


def close_cb(event):
    # TODO: cleanup (save vid? etc)
    if cap:
        cap.release()
        print("released capture")
    root.destroy()
    cv2.destroyAllWindows()
    exit(0)


root.wm_attributes("-fullscreen", 1)
my_gui = Gui(root)

scr_w = root.winfo_screenwidth()
scr_h = root.winfo_screenheight()
# scr_w = 2560
# scr_h = 1440
logging.info(f"screen dimensions: {scr_w}x{scr_h}")

canvas = Canvas(root, width=scr_w, height=scr_h)
canvas.focus_set()
canvas.bind("q", close_cb)
canvas.pack()


def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)


Canvas.create_circle = _create_circle


def create_dot(x, y, r=8, color="red"):
    return canvas.create_circle(x, y, r, fill=color)


def create_trial_dot(x, y):
    dot_big = canvas.create_circle(x, y, 40, fill="red")
    canvas.update()
    time.sleep(dot_config["dot_big_dur"])
    canvas.delete(dot_big)
    dot_small = canvas.create_circle(x, y, 3, fill="red")
    canvas.update()
    time.sleep(dot_config["dot_small_dur"])
    canvas.delete(dot_small)
    canvas.update()


def init_cap():
    global cap
    cap = cv2.VideoCapture(0)

    #fourcc = cv2.VideoWriter_fourcc(*"MJPG") #DIVX, XVID?
    #cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config["cam_res"][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config["cam_res"][1])
    cap.set(cv2.CAP_PROP_FPS, cam_config["cam_fps"])
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    #logging.info(f"""video stats:
    #                               dimensions: {vid_dim},
    #                               fps: {cap.get(cv2.CAP_PROP_FPS)},
    #                               max_duration: {max_dur},
    #                               crop: {crop}""")
    return cap


def test_rec():
    global cap
    cap = init_cap()
    noncrop = (0, cam_config["cam_res"][0], 0, cam_config["cam_res"][1])
    crop = False if cam_config["crop"] == noncrop else True

    while True:
        ret = False
        while not ret:
            ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if crop:
            frame = frame[cam_config["crop"][2]:cam_config["crop"][3],
                          cam_config["crop"][0]:cam_config["crop"][1]]

        cv2.imshow('frame', frame)
        cv2.moveWindow("frame", 1000, 500)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)


def gazevid(event):
    cap = init_cap()

    noncrop = (0, cam_config["cam_res"][0], 0, cam_config["cam_res"][1])
    crop = False if cam_config["crop"] == noncrop else True

    #now = datetime.now().isoformat()
    now = get_enc_time()
    root = "/Users/nzdarsky/code/thesis_bachelor/data/trials"
    os.mkdir(f"{root}/{now}")
    os.mkdir(f"{root}/{now}/img")
    home = f"{root}/{now}"

    with open(f"{home}/gazedots.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["x", "y"])

        logging.info("starting recording...")
        i = 0
        event.set()
        while event.is_set():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                if crop:
                    frame = frame[cam_config["crop"][2]:cam_config["crop"][3],
                                  cam_config["crop"][0]:cam_config["crop"][1]]
            
                w.writerow([dot_x, dot_y])  # TODO: switch?
                cv2.imwrite(f"{home}/img/{i:04d}.jpg", frame)
                i += 1
                #cv2.imshow('frame', frame)# not working?
                #out.write(frame)

    cap.release()
    #out.release()
    logging.info("gazevid finished")


## STARTLIGHTS
def countdown(delay=0.5):
    light_colors = ["red", "orange", "yellow", "green"]
    canvas.update()
    for i in range(len(light_colors)):
        for j in range(1, 6):
            canvas.create_circle(int(scr_w/6 * j), int(scr_h/2), 40,
                                 fill=light_colors[i])
        canvas.update()
        time.sleep(delay)
        canvas.delete("all")
        canvas.update()
    time.sleep(delay)


def linetask(event):
    global dot_x
    global dot_y

    speed_start = line_config["speed_start"]
    padding = line_config["padding"]
    step = line_config["step"]
    #TODO: use tuples?
    dests = []
    left = True

    speed_inc = 0.02
    speed_dec = speed_inc
    speed_start = 1

    for y in range(padding, scr_h+padding, step):
        if left:
            dests.append([padding, y])
            dests.append([scr_w-padding, y])
        else:
            dests.append([scr_w-padding, y])
            dests.append([padding, y])
        left = not left
    dot_x = dests[0][0]
    dot_y = dests[0][0]
    dot = create_dot(dot_x, dot_y)
    canvas.update()
    dest_num = 1
    print("waiting...")
    event.wait()
    print("done waiting!")
    while dest_num < len(dests):
        dest_x = dests[dest_num][0]
        dest_y = dests[dest_num][1]
        #print(f"dests: ({dest_x}, {dest_y})")
        dest_num += 1
        line_dest = canvas.create_line(dot_x, dot_y, dest_x, dest_y)
        canvas.update()
        speed_x = speed_start
        speed_y = speed_start
        maxdist_x = abs(dest_x - dot_x)
        maxdist_y = abs(dest_y - dot_y)
        while (dot_x != dest_x) or (dot_y != dest_y):
            #print(f"{dot_x} != {dest_x} and {dot_y} != {dest_y}: {(dot_x != dest_x) and (dot_y != dest_y)}")
            #sleep(0.01)#0.02 -> 50fps
            time.sleep(1./70.) #  70fps
            #print(f"({dot_x}, {dot_y}) -> ({dest_x}, {dest_y})")
            dist_x = dest_x - dot_x
            dist_y = dest_y - dot_y
            
            # move_x = int(np.sign(dist_x) * speed_start)
            # move_y = int(np.sign(dist_y) * speed_start)#TODO: include screen ratio

            move_x = int((np.sign(dist_x)) * speed_x)
            move_y = int((np.sign(dist_y)) * speed_y)#TODO: include screen ratio

            if abs(dist_x) > int(maxdist_x/2):
                speed_x += speed_inc
            else:
                if speed_x > (speed_start):
                    speed_x -= speed_dec

            if abs(dist_y) > int(maxdist_y/2):
                speed_y += speed_inc
            else:
                if speed_y > (speed_start):
                    speed_y -= speed_dec

            #print(f"move({move_x}, {move_y})")
            #print(f"dist: ({dist_x}, {dist_y})")
            canvas.move(dot, move_x, move_y)
            dot_x += move_x
            dot_y += move_y
            
            canvas.update()
    event.clear()
    logging.info("dottask finished")


def create_attention_catcher(x, y):
    for radius in range(18, 7, -1):
        dot = create_dot(x, y, radius)
        time.sleep(0.07)
        canvas.update()
        canvas.delete(dot)


def gridtask(event):
    global dot_x
    global dot_y

    padding = 50
    step = 100
    scr_w = 2560
    scr_h = 1440
    speed = 5.

    event.wait()
    #### horizontal ####
    dot_x, dot_y = padding, padding
    dest_x = scr_w - padding
    dot = create_dot(dot_x, dot_y)
    direction = 1
    while dot_y <= scr_h - padding:
        # big dot -> decrease size (catch attention)
        create_attention_catcher(dot_x, dot_y)
        if direction == 1:
            dest_x = scr_w - padding
        else:
            dest_x = padding
        while dot_x != dest_x:
            old_x = dot_x
            dot_x += direction * speed
            canvas.move(dot, dot_x - old_x, 0)
            canvas.update()
            time.sleep(1./70.)  # 70fps
        old_y = dot_y
        dot_y += step
        canvas.move(dot, 0, dot_y - old_y)
        direction *= -1
    canvas.delete(dot)
    #### vertical ####
    # dot_x, dot_y = padding, padding
    # dest_y = scr_h - padding
    # dot = create_dot(dot_x, dot_y)

    # direction = 1
    # while dot_x <= scr_w - padding:
        # # big dot -> decrease size (catch attention)
        # create_attention_catcher(dot_x, dot_y)
        # if direction == 1:
            # dest_y = scr_h - padding
        # else:
            # dest_y = padding
        # while dot_y != dest_y:
            # old_y = dot_y
            # dot_y += direction * speed
            # canvas.move(dot, 0, dot_y - old_y)
            # canvas.update()
            # time.sleep(1./70.)  # 70fps
        # old_x = dot_x
        # dot_x += step
        # canvas.move(dot, dot_x - old_x, 0)
        # direction *= -1

    event.clear()
    logging.info("gridtask finished")


test_rec()
canvas.focus_set()
countdown()

logging.info("starting task")
event = threading.Event()
t_gazevid = threading.Thread(target=gazevid, args=(event,))
t_gazevid.start()
linetask(event)
# event.set()
# gridtask(event)

