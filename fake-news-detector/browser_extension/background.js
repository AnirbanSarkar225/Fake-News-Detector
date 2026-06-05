// Create context menu items on installation
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkCredibility",
    title: "🛡️ Verify with TruthShield",
    contexts: ["selection"]
  });
});

// Listener for context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "checkCredibility") {
    // Open the popup or send message
    chrome.storage.local.set({ selectedTextForVerify: info.selectionText }, () => {
      // Open popup window or sidepanel
      chrome.action.openPopup();
    });
  }
});
