export const SCENARIOS = {
  smartPills: {
    name: "Smart Pills vs Placebo",
    groupA: [85, 88, 82, 89, 81], // mean 85.0
    groupB: [92, 95, 91, 90, 96], // mean 92.8
    unit: "pts",
    description: "A pharmaceutical company tests a new 'smart pill' designed to boost cognitive function. Five students take the pill, and five take a sugar pill. These are their scores on a subsequent memory test."
  },
  fertilizer: {
    name: "New Fertilizer vs Old",
    groupA: [12, 14, 13, 15, 14], // mean 13.6
    groupB: [13, 15, 14, 14, 15], // mean 14.2
    unit: "cm",
    description: "A botanist wants to see if a new experimental fertilizer makes plants grow taller than the standard mix. Five seedlings get the old mix, and five get the new one. Here are their heights after two weeks."
  },
  coffee: {
    name: "Coffee vs Decaf",
    groupA: [58, 62, 55, 60, 56], // mean 58.2
    groupB: [65, 71, 62, 68, 64], // mean 66.0
    unit: "wpm",
    description: "Does caffeine actually make you type faster? We asked ten people to transcribe a document. Five were secretly given decaf, while the other five were given strong espresso. Here are their typing speeds."
  }
};
