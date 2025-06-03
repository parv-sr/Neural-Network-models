import random
import json

bedrooms = []
bathrooms = []
sq_ft = []
floors = []
age = []

labels = []

for i in range(1000):
    bedrooms.append(random.randint(1, 6))
    bathrooms.append(random.randint(1, bedrooms[i]))

    total_rooms = bedrooms[i] + bathrooms[i]
    base_sqft = 400 * total_rooms
    noise = random.randint(-100, 100)
    sq_ftage = max(50, base_sqft + noise)
    sq_ft.append(sq_ftage)
    
    if bedrooms[i] > 3 and bedrooms[i] < 5:
        floor_prob = random.random()
        if floor_prob < 0.85:
                floors.append(random.randint(2, 5))
        else:
            floors.append(1)
    elif bedrooms[i] in [5, 6]:
        floors.append(random.randint(3, 5))
    else:
        floors.append(1)
    

    age.append(random.randint(1, 100))

    if (bedrooms[i] >= 3 and bathrooms[i] >= 3) or (sq_ft[i] >= 2000) or (floors[i] >= 2) or (age[i] <= 10):
        prob = random.random()
        if prob < 0.8:
            labels.append(random.randint(8000000, 25000000))
        else:
            labels.append(random.randint(500000, 8000000))
    else:
        labels.append(random.randint(500000, 8000000))



records = {
    "bedrooms":bedrooms,
    "bathrooms":bathrooms,
    "sq_ft":sq_ft,
    "floors":floors,
    "age":age,
    "labels":labels
}

print(records)


try:
    with open("house_cost_dataset.json", "w") as f:
        json.dump(records, f, indent=4)
except Exception as e:
    print("Error while creating file")

print("File created succesfully!")