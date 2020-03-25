import dramatiq
import requests
import sys


@dramatiq.actor
def count_words(url):
    response = requests.get(url)
    count = len(response.text.split(" "))
    print(f"There are {count} words at {url!r}.")


if __name__ == "__main__":
    count_words.send(sys.argv[1])