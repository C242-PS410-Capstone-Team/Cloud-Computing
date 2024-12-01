import { Request, Response, NextFunction } from "express";
import bcrypt from "bcrypt";
import { CreateUserTypes } from "../types/interfaces";
import userDbCollection from "../utils/firestore";
import { v4 as uuidv4 } from "uuid";
import {
  validateEmail,
  validatePassword,
} from "../middlewares/inputValidation";

const registerUser = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  const { email, password } = req.body;

  try {
    // Validate email and password input
    if (!email || !password) {
      res.status(400).json({ message: "Email and password are required." });
      return;
    }

    // Validate email format
    if (!validateEmail(email)) {
      res.status(400).json({ message: "Invalid email format." });
      return;
    }

    // Validate password strength
    const passwordValidationResult = validatePassword(password);
    if (passwordValidationResult !== true) {
      res.status(400).json({ message: passwordValidationResult });
      return;
    }

    // Hash the password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Check if user already exists
    const existingUserSnapshot = await userDbCollection
      .where("email", "==", email)
      .get();
    if (!existingUserSnapshot.empty) {
      res.status(400).json({ message: "User  already exists." });
      return;
    }

    // Create user object
    const newUser: CreateUserTypes = {
      id: uuidv4(),
      email,
      password: hashedPassword,
    };

    // Save user to database with email as document ID
    await userDbCollection.doc(newUser.email).set(newUser);

    // Respond with the created user
    res.status(201).json({
      message: "User  registered successfully.",
      user: {
        id: newUser.id,
        email: newUser.email,
      },
    });
  } catch (error) {
    console.error("Error registering user:", error);
    next(error);
  }
};

export default registerUser;
