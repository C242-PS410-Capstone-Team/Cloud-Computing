// Function to validate email
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// Function to validate password
export const validatePassword = (password: string): true | string => {
  const minLength = 8; // Minimum length for the password
  const hasUpperCase = /[A-Z]/.test(password); // At least one uppercase letter
  const hasLowerCase = /[a-z]/.test(password); // At least one lowercase letter
  const hasNumbers = /\d/.test(password); // At least one number
  const hasSpecialChars = /[!@#$%^&*(),.?":{}|<>]/.test(password); // At least one special character

  if (password.length < minLength) {
    return "Password must be at least 8 characters.";
  }
  if (!hasUpperCase) {
    return "Password must contain at least one uppercase letter.";
  }
  if (!hasLowerCase) {
    return "Password must contain at least one lowercase letter.";
  }
  if (!hasNumbers) {
    return "Password must contain at least one number.";
  }
  if (!hasSpecialChars) {
    return "Password must contain at least one special character.";
  }

  return true;
};
