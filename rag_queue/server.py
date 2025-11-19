from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Query
from .client.rag_client import redis_queue
from .rag_workers.worker import process_queue
app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query:str=Query(..., description="The user query to process")):
    # Use string path - task_worker is at the top level of rag_queue directory
    job = redis_queue.enqueue(process_queue, query)
    return {"job_id": job.get_id(), "status": "queued"}


@app.get("/results")
def job_result(job_id:str = Query(..., description="The job ID to fetch results for")):
    job = redis_queue.fetch_job(job_id= job_id)
    result = job.return_value()
    return {"result": result}