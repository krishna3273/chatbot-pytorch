from tkinter import *
from chat import get_response

GRAY="#ABB2B9"
COLOR="#17202A"
TEXT_COLOR="#EAECEE"

FONT="Helvetica 14"
FONT_BOLD="Helvetica 13 bold"


name="kmt_bot"

class ChatApp:
    def __init__(self):
        self.window=Tk()
        self.setup_main_window()

    def setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False,height=False)
        self.window.configure(width=470,height=550,bg=COLOR)

        head_label=Label(self.window,bg=COLOR,fg=TEXT_COLOR,text="Welcome",font=FONT_BOLD,pady=10)
        head_label.place(relwidth=1)

        line=Label(self.window,width=450,bg=GRAY)
        line.place(relwidth=1,rely=0.07,relheight=0.012)



        self.text_widjet=Text(self.window,width=20,height=2,bg=COLOR,fg=TEXT_COLOR,font=FONT,padx=5,pady=5,wrap=WORD)
        self.text_widjet.place(relheight=0.745,relwidth=1,rely=0.08)
        self.text_widjet.configure(cursor="arrow",state=DISABLED)


        # scrollbar=Scrollbar(self.text_widjet)
        # scrollbar.place(relheight=1,relx=0.974)
        # scrollbar.configure(command=self.text_widjet.yview)

        bottom_label=Label(self.window,bg=GRAY,height=80)
        bottom_label.place(relwidth=1,rely=0.825)

        self.msg_entry=Entry(bottom_label,bg="#2C3E50",fg=TEXT_COLOR,font=FONT)
        self.msg_entry.place(relwidth=0.74,relheight=0.09,rely=0.008,relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>",self.on_enter)

        send_button=Button(bottom_label,text="Send",font=FONT_BOLD,width=20,bg=GRAY,command=lambda:self.on_enter(None))
        send_button.place(relx=0.77,rely=0.008,relheight=0.09,relwidth=0.22)



    def on_enter(self,event):
        msg=self.msg_entry.get()
        self.insert_message(msg,"You")

    def insert_message(self,msg,sender):
        if not msg:
            return
        self.msg_entry.delete(0,END)
        msg=f"{sender}: {msg}\n\n"
        self.text_widjet.configure(state=NORMAL)
        self.text_widjet.insert(END,msg)
        self.text_widjet.configure(state=DISABLED)

        msg_res=f"{name}: {get_response(msg)}\n\n"
        self.text_widjet.configure(state=NORMAL)
        self.text_widjet.insert(END,msg_res)
        self.text_widjet.configure(state=DISABLED)

        self.text_widjet.see(END)

    def run(self):
        self.window.mainloop()

if __name__=="__main__":
    app=ChatApp()
    app.run()