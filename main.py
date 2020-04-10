import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer  #CountVectorizer is a class
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity    #cosine_similarity is a method
from sklearn.metrics.pairwise import sigmoid_kernel
from prettytable import from_csv
from tkinter import *
from PIL import Image,ImageTk
root = Tk()
df = pd.read_csv("phones 2019.csv",encoding='latin1')
#df="phones 2019.csv"
root.geometry("1350x700+0+0")
root.title("RECOMENDATION SYSTEM")
bg_color = "#074463"
title = Label(root, text="SMART DEVICES RECOMENDATION SYSTEM" , bd=12 , relief=GROOVE, bg=bg_color,fg="white", font=("times new roman", 30, "bold") , pady=2).pack(fill=X)
f1=Frame(root,bd=10,relief=GROOVE).place(x=0,y=100,width=1350,height=670)
f1_title = Label(f1,text="SMART PHONES",font="arial 15 bold" , bd=10,relief=GROOVE,bg="light blue",fg="#074463",pady=3).pack(fill=X)
scrol_y = Scrollbar(f1,orient=VERTICAL)
txtarea = Text(f1,bg=bg_color,yscrollcommand=scrol_y.set)
scrol_y.pack(side=RIGHT,fill=Y)
scrol_y.config(command=txtarea.yview)
txtarea.pack(fill=BOTH,expand=1)
f2=LabelFrame(f1,bd=10,relief=GROOVE,text="SMART PHONES ",font=("times new roman",15,"bold"),bg=bg_color,fg="white").place(x=0,y=125,width=900,height=675)
image1 = Image.open("s1.jpg")

image3 = Image.open("s4.jpg")
image4 = Image.open("s5.webp")

image10 = Image.open("s11.jpg")

image12 = Image.open("s13.jpg")

photo1 = ImageTk.PhotoImage(image1)

photo3 = ImageTk.PhotoImage(image3)
photo4 = ImageTk.PhotoImage(image4)

photo10 = ImageTk.PhotoImage(image10)

photo12 = ImageTk.PhotoImage(image12)



image_list = [photo1,photo3,photo4,photo10,photo12]

my_label=Label(f2,image=photo1)
my_label.place(x=15,y=150,width=870,height=635)
def forward(image_number):
    global my_label
    global button_forward
    global button_back
    my_label.grid_forget()
    my_label = Label(image=image_list[image_number - 1])
    button_forward = Button(root, text=">>",font=("arial 28 bold"),bg = bg_color,fg="white",command=lambda: forward(image_number + 1))
    button_back = Button(root, text=">>",font=("arial 28 bold"),bg = bg_color,fg="white", command=lambda: back(image_number - 1))

    if image_number == 5:
        button_forward = Button(root,text=">>",font=("arial 28 bold"),bg = bg_color,fg="white",state=DISABLED)

    button_back.place(x=18, y=450, width=50, height=40)
    button_forward.place(x=830, y=450, width=50, height=40)
    my_label.place(x=15,y=150,width=870,height=635)
def back(image_number):
    global my_label
    global button_forward
    global button_back
    my_label.grid_forget()
    my_label = Label(image=image_list[image_number - 1])
    button_forward = Button(root, text=">>",font=("arial 28 bold"),bg = bg_color,fg="white",command=lambda: forward(image_number + 1))
    button_back = Button(root, text=">>",font=("arial 28 bold"),bg = bg_color,fg="white",command=lambda: back(image_number - 1))

    if image_number == 1:
        button_back = Button(root,text=">>",font=("arial 28 bold"),bg = bg_color,fg="white",state=DISABLED)

    button_back.place(x=18, y=450, width=50, height=40)
    button_forward.place(x=830, y=450, width=50, height=40)
    my_label.place(x=15,y=150,width=870,height=635)

button_back=Button(f2,text="<<",font=("arial 28 bold"),bg = bg_color,fg="white",command=back,state=DISABLED)
button_forward=Button(f2,text=">>",font=("arial 28 bold"),bg = bg_color,fg="white",command=lambda:forward(2))
button_back.place(x=18,y=450,width=50,height=40)
button_forward.place(x=830,y=450,width=50,height=40)


def here_exit():
    sys.exit(0)



#txtarea.insert(END," if u want to exit press Exit Button ")
#button = Button(txtarea,text="EXIT",bg="cadetblue",fg="white",pady=15,width=11,font=("arial 12 bold"),command=here_exit).place(x=1300,y=20)
#s=StringVar()
#entry1 = Entry(txtarea,textvariable=s).pack()
#entry2 = Entry(txtarea,textvariable=s).pack()
#button2 = Button(txtarea,text="click",command=click_me).pack()

f2=Label(txtarea,text="WELCOME",font=("times new roman",20,"bold"),bg=bg_color,fg="white").place(x=600,y=7)
l1=Label(txtarea,text="1.SMART PHONES",font=("times new roman",14,"bold"),bg=bg_color,fg="White").place(x=1000,y=15)
l2=Label(txtarea,text="2.exit",font=("times new roman",14,"bold"),bg=bg_color,fg="white").place(x=1000,y=47)




def get_model_from_index(Index):
    return df[df.Index == Index]["Model"].values[0]


def get_ram_from_index(Index):
    return df[df.Index == Index]["RAM"].values[0]


def get_storage_from_index(Index):
    return df[df.Index == Index]["Storage_capacity"].values[0]


def get_battery_from_index(Index):
    return df[df.Index == Index]["Battery"].values[0]


def get_weight_from_index(Index):
    return df[df.Index == Index]["Weight"].values[0]


def get_index_from_ram(RAM):
    return df[df.RAM == RAM]["Index"].values[0]


def combine_features(row):
    return row['Model'] + " " + row['Battery'] + " " + row['RAM'] + " " + row['Weight']

def details():
    #file = open(df, encoding='latin1')

    features = ["Model", "Battery", "RAM", "Weight"]
    for feature in features:
        df[feature] = df[feature].fillna(' ')

    df["combined_features"] = df.apply(combine_features, axis=1)
    # df["combined_features"]

    cv = CountVectorizer()  # cv is the object of CountVectorizer
    count_matrix = cv.fit_transform(df["combined_features"])  # fit_transform() count the frequency of words in data
    # count_matrix.toarray()

    cosine_sim = cosine_similarity(count_matrix)  # to find the similarities between data
    # cosine_sim

    txtarea.insert(END, " enter the RAM  ")

    # button2 = Button(txtarea,text="click",command=click_me).pack()
    phone_user_like = s2.get()
    try:
        phone_index = int(get_index_from_ram(phone_user_like))
    except Exception:
        print(" Sorry, this info is not present in the file")
        sys.exit(0)

    similar_phones = list(enumerate(cosine_sim[phone_index]))
    # similar_phones
    sorted_similar_phones = sorted(similar_phones, key=lambda x: x[1],
                                   reverse=True)  # sort the second element (x[1]) of tuple in descending(reverse=true) order
    # sorted_similar_phones

    phone_model = [i[0] for i in sorted_similar_phones]
    print("   WEIGHT    BATTERY       RAM                 MODEL")
    for id in range(0, len(phone_model)):
        print(
            f'    {get_weight_from_index(phone_model[id])}     {get_battery_from_index(phone_model[id])}      {get_ram_from_index(phone_model[id])}                 {get_model_from_index(phone_model[id])}')


s2 = StringVar()

def open_window():
    top = Toplevel()
    top.geometry("100x100+1000+40")
    label=Label(top,text="enter ROM ").pack()
    entry = Entry(top , textvariable=s2).pack()
    button = Button(top,text="click",command=details).pack()

def click_me():
    s1 = s.get()
    if (s1 == '1'):
        open_window()

    elif (s1 == '2'):
        sys.exit(0)
    else:
        print("wrong choice")

s=StringVar()
label = Label(txtarea,text="enter the choice",font=("times new roman",14,"bold"),bg=bg_color,fg="white").place(x=1100,y=80)
entry = Entry(txtarea,textvariable=s,font=("times new roman",14,"bold"),bd=5,relief=SUNKEN,width=10).place(x=1100,y=110)
button = Button(txtarea,text="click",bg="cadetblue",fg="white",pady=15,width=11,font=("arial 12 bold"),command=click_me).place(x=1100,y=160)

root.mainloop()
