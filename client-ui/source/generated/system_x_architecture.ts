/** GENERATED FILE: system-x.verify-architecture.v1 */
export type ArchitectureStatus = "PASS" | "FAIL";
export interface ArchitectureResult {
  schema: "system-x.verify-architecture.v1";
  status: ArchitectureStatus;
  package_modules: string[];
  forbidden_domain_imports: string[];
  generated_schema: boolean;
  source_only: true;
}
