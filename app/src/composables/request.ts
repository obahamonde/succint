export const useRequest = <Req, Res>() => {
  const request = async (
    url: string,
    opts?: RequestInit,
    body?: Req,
    refetch?: boolean,
  ) => {
    const options: RequestInit = {
      method: body ? "POST" : "GET",
      headers: {
        "Content-Type": "application/json",
      },
      ...opts,
    };
    const requestOptions: RequestInit = {
      ...options
    }
    const {
      data,
      error,
      isFetching: loading,
    } = await useFetch(url, requestOptions, {
      refetch: refetch || false,
    }).json<Res>();
    return { data, error, loading };
  };
  return request;
};
