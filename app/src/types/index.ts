export type Notification = {
  message: string;
  status: "success" | "error" | "warning" | "info";
};

export type User = {
  id: string;
  email: string | undefined;
  email_verified: boolean | undefined;
  family_name: string | undefined;
  given_name: string | undefined;
  locale: string | undefined;
  name: string;
  nickname: string | undefined;
  picture: string | undefined;
  sub: string;
  updated_at: string | undefined;
};

export type Message = {
  role: "user" | "assistant";
  content: string;
}