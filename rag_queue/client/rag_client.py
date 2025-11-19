from redis import Redis
from rq import Queue


redis_queue = Queue(connection=Redis(host='localhost', port=6379))
