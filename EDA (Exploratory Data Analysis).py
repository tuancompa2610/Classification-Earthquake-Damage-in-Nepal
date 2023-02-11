# Drop leaky features
# Those features with "post" are likely to be leaky featyres since those features happen after the earthquake.
df.drop(columns = ["count_floors_post_eq", "height_ft_post_eq", "condition_post_eq"], inplace = True)
# Create correlation matrix
correlation = df.select_dtypes("number").drop(columns = "severe_damage").corr()
# Plot heatmap of `correlation`
sns.heatmap(correlation)

corr_count_floors_pre_eq = df["severe_damage"].corr(df["count_floors_pre_eq"])
corr_height_ft_pre_eq = df["severe_damage"].corr(df["height_ft_pre_eq"])
print(f"The corelation coefficient between count_floors_pre_eq and severe_damage: {corr_count_floors_pre_eq}")
print(f"The corelation coefficient between height_ft_pre_eq and severe_damage: {corr_height_ft_pre_eq}")

df.drop(columns = "count_floors_pre_eq", inplace = True)

sns.boxplot(x = "severe_damage", y = "plinth_area_sq_ft", data = df)
plt.xlabel("Severe Damage")
plt.ylabel("Plinth Area [sq. ft.]")
plt.title("Plinth Area vs Building Damage")

majority_class_prop, minority_class_prop = df["severe_damage"].value_counts(
    normalize = True)
print(majority_class_prop, minority_class_prop)

# Plot value counts of `"severe_damage"`
df["severe_damage"].value_counts(normalize = True).plot(
	kind = "bar", xlabel = "Class", ylabel = "Relative Frequency", title = "Class Balance"
)

# Create pivot table
foundation_pivot = pd.pivot_table(
	df, index = "foundation_type", values = "severe_damage", aggfunc = np.mean
).sort_values(by = "severe_damage")
foundation_pivot

# Plot bar chart of `foundation_pivot`
foundation_pivot.plot(kind = "barh", legend = None)
plt.axvline(
	majority_class_prop, linestyle = "--", color = "red", label = "majority class"
)
plt.axvline(
	minority_class_prop, linestyle = "--", color = "green", label = "minority class"
)
plt.legend(loc = "lower right")

# Check for high- and low-cardinality categorical features
df.select_dtypes("object").nunique()
