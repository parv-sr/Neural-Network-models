import random
import json

age = []
time = []
weather = []
labels = []

for i in range(1000):
    age.append(random.randint(1, 100))
    time.append(random.randint(0, 24))
    weather.append(random.randint(0, 2))

    if (age[i] >= 16 and age[i] <= 59) and (time[i] >= 6 and time[i] <= 10) or (time[i] >= 14 and time[i] <= 16):
        v = random.random()
        if v > 0.1:
            labels.append(1)
        else:
            labels.append(0)
    else:
        labels.append(0)



records = {
    "age":age,
    "time":time,
    "weather":weather,
    "labels":labels
}


try:
    with open("coffee_preference_dataset.json", "w") as f:
        json.dump(records, f)
except Exception as e:
    print("Error while creating file")

print("File created succesfully!")
