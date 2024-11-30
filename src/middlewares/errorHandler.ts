import { Request, Response, NextFunction } from "express";
import { ErrorResponse } from "../types/interfaces";

const errorHandler = (
  err: ErrorResponse,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const statusCode = err.status || 500;
  const message = err.message || "Internal Server Error";

  res.status(statusCode).json({
    status: statusCode,
    message: message,
  });
};

export default errorHandler;
