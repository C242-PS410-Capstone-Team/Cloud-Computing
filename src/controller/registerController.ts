import { Request, Response } from "express";
import bcrypt from "bcrypt";
import { UserTypes } from "../types/interfaces";
import { randomUUID } from "crypto";
import {
  validateEmail,
  validatePassword,
} from "../middlewares/inputValidation";

const registerUser = async (req: Request, res: Response): Promise<void> => {
  try {
    const { email, password } = req.body;

    // Validate email and password input
    if (!email || !password) {
      res.status(400).json({ message: "Email and password are required" });
    }

    // Validate email
    if (!validateEmail(email)) {
      res.status(400).json({ message: "Invalid email format" });
    }

    // Validate password
    const passwordValidationResult = validatePassword(password);
    if (passwordValidationResult !== true) {
      res.status(400).json({ message: passwordValidationResult });
    }

    // Hash the password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create user object
    const user: UserTypes = {
      id: randomUUID(),
      email,
      password: hashedPassword,
    };

    // Save user to database
    // ---

    // Respond with the created user (excluding password)
    res.status(201).json({
      id: user.id,
      email: user.email,
      message: "User  registered successfully",
    });
  } catch (error) {
    console.error("Error registering user:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};

export default registerUser;
