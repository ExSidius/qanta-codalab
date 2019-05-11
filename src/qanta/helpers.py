import time


def logger(log_text):
	def decorator(f):
		def wrapper(*args, **kwargs):

			start_time = time.time()
			start = time.strftime('%H:%M:%S', time.localtime(start_time))

			print(f'Started {log_text} at {start}...')
			result = f(*args, **kwargs)

			end_time = time.time()
			end = time.strftime('%H:%M:%S', time.localtime(end_time))
			print(f'Finished {log_text} at {end}...')

			print()
			return result
		return wrapper
	return decorator
