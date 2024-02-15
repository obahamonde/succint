export const useEvent = <T>(url: string, callback: (data: T) => any) => {
	const eventSource = new EventSource(url);
	eventSource.onmessage = (event: MessageEvent) => {
		const eventData = JSON.parse(event.data) as T;
		callback(eventData)
		eventSource.addEventListener("done", (event) => {
			eventSource.close();
		}
		);

	}

	return () => {
		eventSource.close();
	};
}

