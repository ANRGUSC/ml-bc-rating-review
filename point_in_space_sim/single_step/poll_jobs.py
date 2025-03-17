import time
import config
import pathlib
import sys
from dotenv import load_dotenv

load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()


def poll_job_status():
    print("Polling job status...")
    time.sleep(300)

    while True:
        try:
            jobs = config.client.fine_tuning.jobs.list()
            jobs_list = []

            for job in jobs:
                jobs_list.append(job)
                if len(jobs_list) >= 3:
                    break

            statuses = [job.status for job in jobs_list[:3]]
            if all(status == 'succeeded' for status in statuses):
                print(f"All three jobs succeeded!")
                time.sleep(20)
                return
            elif any(status in ['failed', 'cancelled'] for status in statuses):
                print(
                    f"Error: At least one failed: {', '.join(set(statuses))}")
                sys.exit(1)
            time.sleep(90)
        except Exception as e:
            print(f"Error occurred: {e}")
            sys.exit(1)


if __name__ == "__main__":
    poll_job_status()
