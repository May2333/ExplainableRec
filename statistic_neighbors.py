from tqdm import tqdm

user_ne_items_path = "saved/Electronics_5/user_ne_items.txt"
user_ne_users_path = "saved/Electronics_5/user_ne_30.txt"
item_ne_users_path = "saved/Electronics_5/item_ne_users.txt"
item_ne_items_path = "saved/Electronics_5/item_ne.txt"

"""
    user:
        sum_user_ne_items
        sum_user_ne_users
        avg_user_ne_items
        avg_user_ne_users
    item:
        sum_item_ne_users
        sum_item_ne_items
        avg_item_ne_users
        avg_item_ne_items
"""
sum_user_ne_items = 0
sum_user_ne_users = 0
avg_user_ne_items = 0
avg_user_ne_users = 0
sum_item_ne_users = 0
sum_item_ne_items = 0
avg_item_ne_users = 0
avg_item_ne_items = 0

with open(user_ne_items_path, 'r', encoding="utf-8") as fin:
    lines = fin.readlines()
    num_users = len(lines)

    for line in tqdm(lines):
        num_ne = len(line.split()) - 1
        sum_user_ne_items += num_ne

    avg_user_ne_items = sum_user_ne_items / num_users

with open(user_ne_users_path, 'r', encoding="utf-8") as fin:
    lines = fin.readlines()
    num_users = len(lines)

    for line in tqdm(lines):
        num_ne = len(line.split()) - 1
        sum_user_ne_users += num_ne

    avg_user_ne_users = sum_user_ne_users / num_users

with open(item_ne_users_path, 'r', encoding="utf-8") as fin:
    lines = fin.readlines()
    num_items = len(lines)

    for line in tqdm(lines):
        num_ne = len(line.split()) - 1
        sum_item_ne_users += num_ne

    avg_item_ne_users = sum_item_ne_users / num_items

with open(item_ne_items_path, 'r', encoding="utf-8") as fin:
    lines = fin.readlines()
    num_items = len(lines)

    for line in tqdm(lines):
        num_ne = len(line.split()) - 1
        sum_item_ne_items += num_ne
    
    avg_item_ne_items = sum_item_ne_items / num_items

print("""
    user:
        sum_user_ne_items = {}
        sum_user_ne_users = {}
        avg_user_ne_items = {}
        avg_user_ne_users = {}
    item:
        sum_item_ne_users = {}
        sum_item_ne_items = {}
        avg_item_ne_users = {}
        avg_item_ne_items = {}
    """.format(
            sum_user_ne_items,
            sum_user_ne_users,
            avg_user_ne_items,
            avg_user_ne_users,
            sum_item_ne_users,
            sum_item_ne_items,
            avg_item_ne_users,
            avg_item_ne_items
    )
)