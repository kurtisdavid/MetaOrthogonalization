import tkinter as tk
from PIL import Image, ImageTk
import os
import glob

class Example(tk.Frame):
    def __init__(self, parent, dataset="idenprof/train"):
        tk.Frame.__init__(self, parent)
        self.setup_images(dataset)
        
        self.male = tk.Button(self, text="Male", command = self.log_m)
        self.female = tk.Button(self, text="Female", command = self.log_f)
        self.both = tk.Button(self, text="Both", command = self.log_both)

        # lay the widgets out on the screen.
        self.both.pack(side="right")
        self.male.pack(side="right")
        self.female.pack(side="right")
    
    def log_m(self):
        self.log('M')

    def log_f(self):
        self.log('F')

    def log_both(self):
        self.log('B')

    def log(self, label):
        with open(os.path.join(self.root,'labels.txt'),'a') as f:
            f.write(self.curr_image + " " + str(label) + '\n')
        self.render_next_image()

    def setup_images(self, root):
        self.root = root
        self.done = set()
        with open(os.path.join(root,'labels.txt')) as f:
            for line in f:
                file_ = line.split(" ")[0]
                self.done.add(file_)

        dirs = [x for x in glob.glob(os.path.join(root,'*')) if '.txt' not in x]
        self.images = []
        for dir_ in dirs:
            images = glob.glob(os.path.join(dir_,'*.jpg'))
            images = [x for x in images if x not in self.done]
            self.images.extend(images)
        self.render_next_image()

    def render_next_image(self):
        self.curr_image = self.images.pop()
        load = Image.open(self.curr_image).resize((400,400))
        render = ImageTk.PhotoImage(load)
        img = tk.Label(self, image=render)
        img.image = render
        img.place(x=0, y=0) 

# if this is run as a program (versus being imported),
# create a root window and an instance of our example,
# then start the event loop

if __name__ == "__main__":
    root = tk.Tk()
    Example(root).pack(fill="both", expand=True)
    root.geometry("500x500")
    root.mainloop()
