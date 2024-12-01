import { Request, Response, NextFunction } from "express";
import bcrypt from "bcrypt";
import { UserTypes } from "../types/interfaces";
import { v4 as uuidv4 } from "uuid";
import userDbCollection from "../utils/firestore";
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
      res.status(400).json({ message: "Email and password are required" });
      return;
    }

    // Validate email format
    if (!validateEmail(email)) {
      res.status(400).json({ message: "Invalid email format" });
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
      res.status(400).json({ message: "User  already exists" });
      return next();
    }

    // Create user object
    const newUser: UserTypes = {
      id: uuidv4(),
      email,
      password: hashedPassword,
    };

    // Save user to database
    await userDbCollection.doc(newUser.id).set(newUser);

    // Respond with the created user
    res.status(201).json({
      id: newUser.id,
      email: newUser.email,
      message: "User  registered successfully",
    });
  } catch (error) {
    console.error("Error registering user:", error);
    return next(error); // Use next for centralized error handling
  }
};

export default registerUser;
