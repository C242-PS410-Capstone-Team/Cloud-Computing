import { Request, Response } from "express";
import bcrypt from "bcrypt";
import userDbCollection from "../utils/firestore";
import { LoginUserTypes } from "../types/interfaces"; // Assuming you have a User interface defined

const loginUser = async (req: Request, res: Response): Promise<void> => {
  const { email, password } = req.body;

  // Input validation
  if (!email || !password) {
    res.status(400).json({ message: "Email and password are required." });
    return;
  }

  try {
    // Retrieve the user document
    const userDocument = await userDbCollection.doc(email).get();

    // Check if the user exists
    if (!userDocument.exists) {
      res.status(404).json({ message: "User  not found." });
      return;
    }

    // Get the user data
    const userData = userDocument.data() as LoginUserTypes;
    if (!userData || !userData.password) {
      res.status(500).json({ message: "User  data is corrupted." });
      return;
    }

    // Compare the provided password with the stored hashed password
    const isPasswordValid = await bcrypt.compare(password, userData.password);

    // Check if the password is valid
    if (!isPasswordValid) {
      res.status(401).json({ message: "Invalid password." });
      return;
    }

    // If password is valid, respond with user info (excluding sensitive data)
    res.status(200).json({
      message: "Login successful.",
      user: {
        id: userData.id,
        email: userData.email,
      },
    });
  } catch (error) {
    console.error("Error during login:", error);
    res.status(500).json({ message: "Internal server error." });
  }
};

export default loginUser;
