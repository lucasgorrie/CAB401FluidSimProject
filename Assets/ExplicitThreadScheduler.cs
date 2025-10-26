using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

internal class ExplicitThreadScheduler : TaskScheduler {

	private readonly LinkedList<Task> _tasks = new LinkedList<Task>();
	private int _delegatesQueuedOrRunning = 0;
	private readonly int _maxDegreeOfParallelism;

	public ExplicitThreadScheduler(int maxDoP) {
		_maxDegreeOfParallelism = maxDoP;
	}

	protected override IEnumerable<Task> GetScheduledTasks() {
		bool lockTaken = false;
		try {
			Monitor.TryEnter(_tasks, ref lockTaken);
			if (lockTaken) return _tasks;
			else throw new NotSupportedException();
		}
		finally {
			if (lockTaken) Monitor.Exit(_tasks);
		}
	}

	protected override void QueueTask(Task task) {
		lock (_tasks) {
			_tasks.AddLast(task);

			// Start a new thread if we're below max DoP
			if (_delegatesQueuedOrRunning < _maxDegreeOfParallelism) {
				_delegatesQueuedOrRunning++;
				StartWorkerThread();
			}
		}
	}

	private void StartWorkerThread() {
		new Thread(() => {
			while (true) {
				Task task;
				lock (_tasks) {
					if (_tasks.Count == 0) {
						_delegatesQueuedOrRunning--;
						break;
					}
					task = _tasks.First.Value;
					_tasks.RemoveFirst();
				}

				TryExecuteTask(task); // Execute on this explicit thread
			}
		})
		{
			IsBackground = true
		}.Start();
	}

	protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued) {
		// Prevent inlining; we want explicit threads only
		return false;
	}
}
