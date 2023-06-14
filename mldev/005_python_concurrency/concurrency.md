Python threading and concurrency practice
================

In this post, I’m learning the basics of python’s threading and
concurrency libraries (like asyncio).

This module is not strictly related to ML

## Threading library

Let’s understand the basics of how the Python threading library works.

Consider the code below

``` python
import time

start = time.perf_counter()

def worker():
    time.sleep(1)
    print(f'done with work')

worker()

end = time.perf_counter()
print(f'Total time {round(end - start, 2)} secs')
```

    done with work
    Total time 1.0 secs

In this code, our `worker` function is a simple synchronous function
that takes 1s to finish.

And the entire execution of the program is also synchronous. So the
entire program finishes in 1s.

If we were to call `worker` three times, we’d expect the program to take
3 seconds.

``` python
import time

start = time.perf_counter()

def worker():
    time.sleep(1)
    print(f'done with work')

worker()
worker()
worker()

end = time.perf_counter()
print(f'Total time {round(end - start, 2)} secs')
```

    done with work
    done with work
    done with work
    Total time 3.01 secs

Clearly, we can do better than this. Each piece of work takes 1 second,
but we are waiting for one to finish before starting the other one.

Let’s see how we can use the threading library to resolve this.

``` python
import threading
import time

start = time.perf_counter()

def worker():
    time.sleep(1)
    print(f'done with work')

# create three threads
t1 = threading.Thread(target=worker)
t2 = threading.Thread(target=worker)
t3 = threading.Thread(target=worker)

# when the threads are started, they execute independently
# without waiting for each other
t1.start()
t2.start()
t3.start()

# wait for the threads to finish before moving on.
t1.join()
t2.join()
t3.join()

end = time.perf_counter()
print(f'Total time {round(end - start, 2)} secs')
```

    done with workdone with work

    done with work
    Total time 1.01 secs

This did make the entire execution take 1 second like we expect.

Aside:

Interestingly, the output of the program mixes up outputs from different
threads, which is an indicator that `print` to stdout this context is
not thread-safe. Also interestingly, this mixing up of output happens in
a Jupyter notebook environment but not if you were to run this as a
standalone shell program.

The reason for this is explained in [this stackoverflow
post](https://stackoverflow.com/questions/42867866/what-makes-python3s-print-function-thread-safe).

> When interactive, stdout and stderr streams are line-buffered.
> Otherwise, they are block-buffered like regular text files. You can
> override this value with the -u command-line option.

> So its really the combination of interactive mode and sys.stderr that
> is responsible for the behaviour of the print function \[…\]

## ThreadPoolExecutor

So far, we’ve been executing threads that print stuff.

Let’s see how we can return values from our threads to our main program.
This is [pretty annoying to do with the raw `threading`
module](https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python),
so we’ll use a new API in Python 3.2 from the `concurrent.futures`
module.

``` python
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

random.seed(0)

start = time.perf_counter()

# our worker function now returns a value
def worker(duration: float):
    time.sleep(duration)
    return(f'done with work for {duration}')

with ThreadPoolExecutor() as executor:
    futures = []
    for _ in range(3):
        duration = random.choice(range(1, 5))
        futures.append(executor.submit(worker, duration))

    for f in as_completed(futures):
        print(f.result())

end = time.perf_counter()
print(f'Total time {round(end - start, 2)} secs')
```

    done with work for 1
    done with work for 4
    done with work for 4
    Total time 4.01 secs

This is pretty cool. If our program has work that involves a lot of
waiting (like making a network request), we can do distribute such work
across multiple threads, and get their outputs in our main program.

`ThreadPoolExecutor.submit` returns a `Future` and when we call
`.result()` on it, we block until the future is resolved. A `Future` is
pretty similar to a `Promise` in Javascript if you’ve come across it
before.

It is very common that you want to run the same piece of code across
many threads. There is an even nicer syntax for this -
`ThreadPoolExecutor.map`

``` python
import random
import time
from concurrent.futures import ThreadPoolExecutor

random.seed(0)

start = time.perf_counter()

def worker(duration: float):
    time.sleep(duration)
    return(f'done with work for {duration}')

with ThreadPoolExecutor() as executor:
    data = [random.choice([1, 2, 3, 4]) for _ in range(3)]
    results = executor.map(worker, data)
    for r in results:
        print(r)

end = time.perf_counter()
print(f'Total time {round(end - start, 2)} secs')
```

    done with work for 4
    done with work for 4
    done with work for 1
    Total time 4.01 secs

The output is slightly different because `ThreadPoolExecutor.map`
returns results in the order that they were passed in.

The same API can also be used to distribute work across multiple
processes rather than multiple threads in the same process, which has
the advantage of sidestepping the Python global interpreter lock.

``` python
import random
import time
from concurrent.futures import ProcessPoolExecutor


def worker(duration: float):
    time.sleep(duration)
    return f"done with work for {duration}"


# we moved all our code under this if statement
# to enable this code to run across multiple processes
if __name__ == "__main__":
    random.seed(0)

    start = time.perf_counter()

    with ProcessPoolExecutor() as exe:
        data = [random.choice([1, 2, 3, 4]) for _ in range(3)]
        results = exe.map(worker, data)
        for r in results:
            print(r)

    end = time.perf_counter()
    print(f"Total time {round(end - start, 2)} secs")
```

This code doesn’t run in a Juptyer notebook, but it does run nicely as a
standalone shell program.

We added a conditional `if __name__ == '__main__'` because the
`ProcessPoolExecutor` should be triggering a new execution only when the
parent program is running, not when each of the `worker` programs are
running.

## asyncio

There is an alternate way of writing concurrent code in Python, which is
using the `asyncio` module and `async/await` syntax.

This syntax was introduced in Python 3.4.

There is a fantastic comparision of `asyncio` vs `threading` /
`concurrent.futures` in [this blog
post](http://masnun.rocks/2016/10/06/async-python-the-different-forms-of-concurrency/).

Let’s see how we’d write similar programs using asyncio.

When we mark functions `async` and call them, they don’t return their
result anymore

``` python
import asyncio

async def foo():
    return 'hello'

print(f"foo={foo()}")
```

    foo=<coroutine object foo at 0x109b09a80>

    /var/folders/v4/zcl4mdss439gswj_l5kb2mr00000gn/T/ipykernel_12088/2004336360.py:6: RuntimeWarning: coroutine 'foo' was never awaited
      print(f"foo={foo()}")
    RuntimeWarning: Enable tracemalloc to get the object allocation traceback

We see that `foo()` doesn’t return `'hello'` like we expect

Instead, it returns a coroutine object. We can have to execute this
coroutine object in order to get the result back.

We can do this by using the `await` keyword.

``` python
import asyncio

async def foo():
    return 'hello'

print(f"foo={await foo()}")
```

    foo=hello

This works nicely, but only because we’re running in a Jupyter notebook.

If we ran this in a standalone Python shell, we’d need slightly
different syntax.

``` python
import asyncio

async def foo():
    return 'hello'

async def main():
    await foo()

# This syntax doesn't work in a Jupyter notebook cell because 
# Jupyter is already running an asyncio event loop.
#
# but it works in a standalone python shell
asyncio.run(main())
```

Now that we see what a coroutine is, let’s write some code that waits
for a long time to see how we’d write such code using `asyncio`.

``` python
import asyncio
import time


def foo():
    time.sleep(1)
    return 'hello'

async def main():
    start = time.perf_counter()
    
    # these are plain old synchronous function calls
    foo()
    foo()
    foo()

    end = time.perf_counter()

    print(f'Total time {round(end - start, 2)} secs')


await main()
```

The code above isn’t anything fancy. `main()` makes three synchronous
function calls to `foo()` and takes a total of 3 seconds to run.

To make the waiting happen concurrently, we’ll switch from using
`time.sleep` to `asyncio.sleep`, and add some `async` / `await`
keywords.

``` python
import asyncio
import time


async def foo():
    # wait for the sleep to finish before executing the next commmand
    await asyncio.sleep(1)
    return f'done sleeping 1 sec'

async def main():
    start = time.perf_counter()
    
    # these will still behave like plain old synchronous function calls
    # we will finish one before moving on to the next one
    print(await foo())
    print(await foo())
    print(await foo())

    end = time.perf_counter()

    print(f'Total time {round(end - start, 2)} secs')


await main()
```

We added some async looking code, but our overall runtime hasn’t changed
yet.

This is because we are waiting for each `foo()` coroutine to finish
before starting the next one.

Instead, we want to start all of them, and then wait for all of them to
finish.

This is where the concept of a `task` comes in handy.

``` python
import asyncio
import time


async def foo():
    # wait for the sleep to finish before executing the next commmand
    await asyncio.sleep(1)
    return f'done sleeping 1 sec'

async def main():
    start = time.perf_counter()
    
    # we run each coroutine in a task
    # this means that anytime one of them pauses, execution
    # proceeds on another one

    t1 = asyncio.create_task(foo())
    t2 = asyncio.create_task(foo())
    t3 = asyncio.create_task(foo())

    print(await t1)
    print(await t2)
    print(await t3)

    end = time.perf_counter()

    print(f'Total time {round(end - start, 2)} secs')


await main()
```

Nice, this lets us run long-running code and get return values back.

Looking at both `concurrent.futures` and `asyncio`, **I’d prefer using
`asyncio` when we’re doing different IO operations in sequence**.

For example, if you’re writing a function that hits your database, then
a third party API, then a cloud-storage service, you could write this
code by chaining together three `await` calls. Writing such code using
the `threading` library wouldn’t be so nice, because creating a new
thread for each different piece of work would require more code.

### asyncio Queues

`asyncio.Queue` is a handy module for sharing data between different
coroutines

Let’s recreate the same example using a queue

``` python
import asyncio
import time


async def foo(name, q):
    while True:
        duration = await q.get()

        await asyncio.sleep(duration)
        q.task_done()

        print(f'{name} slept for {duration} secs')

async def main():
    q = asyncio.Queue()
    start = time.perf_counter()
    
    # create three worker tasks
    t1 = asyncio.create_task(foo('worker-1', q))
    t2 = asyncio.create_task(foo('worker-2', q))
    t3 = asyncio.create_task(foo('worker-3', q))


    # add 10 items to the queue
    for _ in range(10):
        duration = random.choice(range(1, 5))
        q.put_nowait(duration)

    # wait for all items from the queue to be fully processed
    await q.join()

    # stop executing the workers
    t1.cancel()
    t2.cancel()
    t3.cancel()
    
    asyncio.gather(t1, t2, t3, return_exceptions=True)
    end = time.perf_counter()
    print(f'Total time {round(end - start, 2)} secs')


await main()
```
