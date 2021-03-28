import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from tkinter import *
from tkinter import ttk
SEMAIL=""
PASSWORD=""
regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
def send_mail_with_text(subject,file):
    msg = EmailMessage()
    f = open("emails.txt", "r")
    recipients=[]
    for i in f:
        recipients.append(i.rstrip("\n"))
    msg['Subject'] = subject
    msg['From'] = SEMAIL
    msg['To'] = ",".join(recipients)
    msg['Date'] = formatdate(localtime = True)

    with open(file, 'rb') as f:
        file_data = f.read()
    msg.add_attachment(file_data, maintype="application", subtype="plain", filename=file)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(SEMAIL, PASSWORD)
        smtp.send_message(msg)




tkWindow = Tk()
tkWindow.geometry('400x150')
tkWindow.title('Welcome')

semailLabel = Label(tkWindow,text="Email").grid(row=0, column=0)
semail = StringVar()
semailEntry = Entry(tkWindow, textvariable=semail).grid(row=0, column=1)


passwordLabel = Label(tkWindow,text="Password").grid(row=2, column=0)
password = StringVar()
passwordEntry = Entry(tkWindow, textvariable=password, show='*').grid(row=2, column=1)

SEMAIL=""
PASSWORD=""
def close_window ():

    global SEMAIL,PASSWORD
    SEMAIL=semail.get()
    PASSWORD=password.get()
    f = open("teacher_db.txt", "r")
    db=f.read()
    l=db.split("\n")
    dict={}
    for i in l:
        dict[i.split(" ")[0]]=i.split(" ")[1]
    if(SEMAIL in dict and dict[SEMAIL]==PASSWORD):
        tkWindow.destroy()

    else:
        msg="Invalid Login Credentials"
        popup = Tk()
        popup.wm_title("!")
        label = ttk.Label(popup, text=msg, font=("Helvetica", 10))
        label.pack(side="top", fill="x", pady=10)
        B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
        B1.pack()
        popup.mainloop()


sendButton = Button(tkWindow, text="Send", command=close_window).grid(row=3, column=0)

tkWindow.mainloop()
subject="Test"
file="questions.txt"
send_mail_with_text(subject, file)
