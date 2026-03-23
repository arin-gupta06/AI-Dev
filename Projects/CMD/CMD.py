#creating own cmd version
import os
FILE_NAME = "tasks.txt"
# load tasks

def load_tasks():
    tasks = {}
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, "r") as file:
            for line in file:
                task_id, title, status = line.strip().split(" | ")
                tasks[int(task_id)] = {"title": title, "status": status}
    return tasks
# save the tasks
def save_tasks(tasks):
    with open (FILE_NAME, "w") as file:
        for task_id, task in tasks.items():
            file.write(f"{task_id} | {task["title"]} | {task["status"]} \n")
            
            
# Add new tasks
def add_tasks(tasks):
    task = input("Enter the task which you want to add: ")
    task_id = max(tasks.keys(), default = 0) + 1
    tasks[task_id] = {"title": task, "status": "incomplete"}
    save_tasks(tasks)

# delete task
def delete_task(tasks):
    if not tasks:
        print("Tasks list is empty")
    else:
        found = False
        deleted_id = int(input("Enter the ID which you want to delete: "))
        for task_id, task in tasks.items():
            if deleted_id == task_id:
                deleted_task = tasks.pop(deleted_id)
                found = True
                print(f"Task {deleted_id} - {deleted_task["title"]} got deleted.")
                save_tasks(tasks)
                return
        if not found:
            print("task not found")

# mark as complete

def mark_as_complete(tasks):
    if not tasks:
        print("Task list is empty")
    else:
        found = False
        mark_as_completed_id = int(input("Enter the ID which have been completed: "))
        for task_id, task in tasks.items():
            if mark_as_completed_id == task_id:
                task["status"] = "completed"
                found = True
                save_tasks(tasks)
                break
            
        if not found:
            print("Task not found")
            
# View all tasks
def display_tasks(tasks):
    if not tasks:
        print("Tasks list is empty")
    else:
        for task_id, task in tasks.items():
            print(f"{task_id} | {task["title"]} - {task["status"]} \n")
            
def main():
    tasks = load_tasks()
    
    while True:
        print("\n=== Main Menu ===")
        print("1. View All tasks \n")
        print("2. Add new task \n")
        print("3. Delete task \n")
        print("4. Mark as complete \n")
        print("5. Exit \n")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            display_tasks(tasks)
        elif choice == "2":
            add_tasks(tasks)
        elif choice == "3":
            delete_task(tasks)
        elif choice == "4":
            mark_as_complete(tasks)
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()