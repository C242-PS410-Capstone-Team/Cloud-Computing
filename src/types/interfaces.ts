export interface CreateUserTypes {
  id: string;
  email: string;
  password: string;
}

export interface LoginUserTypes {
  id: string;
  email: string;
  password: string;
}

export interface ErrorResponse {
  status: number;
  message: string;
}
