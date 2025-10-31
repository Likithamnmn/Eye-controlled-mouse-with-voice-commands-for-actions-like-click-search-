import React from "react";

export default function Switch({ checked = false, onCheckedChange }) {
  const handleClick = () => {
    if (onCheckedChange) onCheckedChange(!checked);
  };

  return (
    <button
      onClick={handleClick}
      className={`w-12 h-6 flex items-center rounded-full p-1 transition ${
        checked ? "bg-green-500" : "bg-gray-400"
      }`}
    >
      <div
        className={`bg-white w-4 h-4 rounded-full shadow-md transform transition ${
          checked ? "translate-x-6" : ""
        }`}
      />
    </button>
  );
}
